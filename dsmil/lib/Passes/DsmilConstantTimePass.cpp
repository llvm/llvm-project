/**
 * @file DsmilConstantTimePass.cpp
 * @brief DSLLVM Constant-Time Enforcement Pass (v1.2 - Feature 10.4)
 *
 * Enforces constant-time execution for cryptographic code to prevent
 * timing side-channel attacks. Functions marked with dsmil_secret
 * must not have:
 * - Secret-dependent branches (if/switch on secret data)
 * - Secret-dependent memory access (array indexing by secrets)
 * - Variable-time instructions (div/mod on secrets)
 *
 * This pass implements compiler-level constant-time support as described
 * in the LLVM security article:
 * https://securityboulevard.com/2025/11/constant-time-support-lands-in-llvm-protecting-cryptographic-code-at-the-compiler-level/
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/CFG.h"
#include <set>
#include <map>
#include <string>

#define DEBUG_TYPE "dsmil-constant-time"

using namespace llvm;

// Command-line options
static cl::opt<bool> EnableCTCheck(
    "dsmil-ct-check",
    cl::desc("Enable constant-time enforcement for dsmil_secret functions"),
    cl::init(true));

static cl::opt<bool> CTCheckStrict(
    "dsmil-ct-check-strict",
    cl::desc("Strict mode: treat warnings as errors"),
    cl::init(false));

static cl::opt<bool> CTCheckDisableDiv(
    "dsmil-ct-check-no-div",
    cl::desc("Disallow division/modulo on secret data"),
    cl::init(true));

static cl::opt<std::string> CTCheckOutput(
    "dsmil-ct-check-output",
    cl::desc("Output path for constant-time violations report"),
    cl::init("ct-violations.json"));

namespace {

enum class ViolationType {
  SecretDependentBranch,
  SecretDependentMemory,
  VariableTimeInstruction,
  SecretLeak
};

struct CTViolation {
  ViolationType Type;
  std::string FunctionName;
  std::string InstructionDesc;
  unsigned LineNumber;
  std::string Message;

  CTViolation(ViolationType T, StringRef FuncName, const Instruction *I, StringRef Msg)
      : Type(T), FunctionName(FuncName.str()), InstructionDesc(""), LineNumber(0), Message(Msg.str()) {
    if (I) {
      raw_string_ostream OS(InstructionDesc);
      I->print(OS);

      if (const DebugLoc &DL = I->getDebugLoc()) {
        LineNumber = DL.getLine();
      }
    }
  }
};

/**
 * Constant-Time Enforcement Pass
 */
class DsmilConstantTimePass : public PassInfoMixin<DsmilConstantTimePass> {
private:
  bool Enabled;
  bool StrictMode;
  bool DisableDiv;
  std::string OutputPath;

  // Track secret-tainted values
  std::set<const Value *> SecretValues;

  // Violations found
  std::vector<CTViolation> Violations;

  /**
   * Check if function has dsmil_secret attribute
   */
  bool hasDsmilSecretAttr(const Function &F) const {
    return F.hasFnAttribute("dsmil_secret");
  }

  /**
   * Check if parameter has dsmil_secret attribute
   */
  bool isSecretParameter(const Argument *Arg) const {
    if (!Arg)
      return false;

    const Function *F = Arg->getParent();
    if (!F)
      return false;

    // Check for parameter-specific secret attribute
    // (This would require Clang to attach parameter attributes)
    // For now, if the function is marked secret, all pointer/integer params are tainted
    if (hasDsmilSecretAttr(*F)) {
      Type *Ty = Arg->getType();
      return Ty->isPointerTy() || Ty->isIntegerTy();
    }

    return false;
  }

  /**
   * Initialize secret tainting from function parameters and globals
   */
  void initializeSecretTainting(Function &F) {
    SecretValues.clear();

    // If function is marked dsmil_secret, taint all sensitive parameters
    if (hasDsmilSecretAttr(F)) {
      for (Argument &Arg : F.args()) {
        if (isSecretParameter(&Arg)) {
          SecretValues.insert(&Arg);
          LLVM_DEBUG(dbgs() << "  Tainting parameter: " << Arg.getName() << "\n");
        }
      }
    }

    // Check for loads from secret globals
    Module *M = F.getParent();
    for (GlobalVariable &GV : M->globals()) {
      if (GV.hasAttribute("dsmil_secret")) {
        SecretValues.insert(&GV);
        LLVM_DEBUG(dbgs() << "  Tainting global: " << GV.getName() << "\n");
      }
    }
  }

  /**
   * Propagate secret taint through SSA graph
   */
  void propagateSecretTaint(Function &F) {
    bool Changed = true;
    unsigned Iterations = 0;
    const unsigned MaxIterations = 100;

    while (Changed && Iterations < MaxIterations) {
      Changed = false;
      ++Iterations;

      for (BasicBlock &BB : F) {
        for (Instruction &I : BB) {
          bool WasTainted = SecretValues.count(&I) > 0;
          bool ShouldBeTainted = false;

          // Check if any operand is secret
          for (Use &U : I.operands()) {
            if (SecretValues.count(U.get()) > 0) {
              ShouldBeTainted = true;
              break;
            }
          }

          // Propagate through loads
          if (auto *LI = dyn_cast<LoadInst>(&I)) {
            if (SecretValues.count(LI->getPointerOperand()) > 0) {
              ShouldBeTainted = true;
            }
          }

          if (ShouldBeTainted && !WasTainted) {
            SecretValues.insert(&I);
            Changed = true;
          }
        }
      }
    }

    LLVM_DEBUG(dbgs() << "  Secret taint propagation converged after " << Iterations << " iterations\n");
    LLVM_DEBUG(dbgs() << "  " << SecretValues.size() << " values tainted as secret\n");
  }

  /**
   * Check if value is tainted as secret
   */
  bool isSecretValue(const Value *V) const {
    return SecretValues.count(V) > 0;
  }

  /**
   * Check for secret-dependent branches
   */
  void checkSecretBranches(Function &F) {
    for (BasicBlock &BB : F) {
      auto *BI = dyn_cast<BranchInst>(BB.getTerminator());
      if (!BI || !BI->isConditional())
        continue;

      Value *Condition = BI->getCondition();
      if (isSecretValue(Condition)) {
        std::string Msg = "Secret-dependent branch: branching on secret value '" +
                          Condition->getName().str() + "'";
        Violations.emplace_back(ViolationType::SecretDependentBranch,
                                F.getName(), BI, Msg);
      }
    }

    // Check switch instructions
    for (BasicBlock &BB : F) {
      auto *SI = dyn_cast<SwitchInst>(BB.getTerminator());
      if (!SI)
        continue;

      Value *Condition = SI->getCondition();
      if (isSecretValue(Condition)) {
        std::string Msg = "Secret-dependent switch: switching on secret value '" +
                          Condition->getName().str() + "'";
        Violations.emplace_back(ViolationType::SecretDependentBranch,
                                F.getName(), SI, Msg);
      }
    }

    // Check select instructions (ternary operator)
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        auto *Sel = dyn_cast<SelectInst>(&I);
        if (!Sel)
          continue;

        Value *Condition = Sel->getCondition();
        if (isSecretValue(Condition)) {
          std::string Msg = "Secret-dependent select: selecting based on secret condition '" +
                            Condition->getName().str() + "'";
          Violations.emplace_back(ViolationType::SecretDependentBranch,
                                  F.getName(), Sel, Msg);
        }
      }
    }
  }

  /**
   * Check for secret-dependent memory accesses
   */
  void checkSecretMemoryAccess(Function &F) {
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        // Check GetElementPtr with secret indices
        if (auto *GEP = dyn_cast<GetElementPtrInst>(&I)) {
          for (auto It = GEP->idx_begin(); It != GEP->idx_end(); ++It) {
            if (isSecretValue(*It)) {
              std::string Msg = "Secret-dependent memory access: array indexing with secret value '" +
                                (*It)->getName().str() + "'";
              Violations.emplace_back(ViolationType::SecretDependentMemory,
                                      F.getName(), GEP, Msg);
            }
          }
        }

        // Check loads with secret addresses
        if (auto *LI = dyn_cast<LoadInst>(&I)) {
          // Allow loads from secret pointers (reading secret data is OK)
          // But disallow computing the address using secret values
          if (auto *GEP = dyn_cast<GetElementPtrInst>(LI->getPointerOperand())) {
            for (auto It = GEP->idx_begin(); It != GEP->idx_end(); ++It) {
              if (isSecretValue(*It)) {
                std::string Msg = "Secret-dependent load: loading from address computed with secret index '" +
                                  (*It)->getName().str() + "'";
                Violations.emplace_back(ViolationType::SecretDependentMemory,
                                        F.getName(), LI, Msg);
              }
            }
          }
        }

        // Check stores with secret addresses
        if (auto *SI = dyn_cast<StoreInst>(&I)) {
          if (auto *GEP = dyn_cast<GetElementPtrInst>(SI->getPointerOperand())) {
            for (auto It = GEP->idx_begin(); It != GEP->idx_end(); ++It) {
              if (isSecretValue(*It)) {
                std::string Msg = "Secret-dependent store: storing to address computed with secret index '" +
                                  (*It)->getName().str() + "'";
                Violations.emplace_back(ViolationType::SecretDependentMemory,
                                        F.getName(), SI, Msg);
              }
            }
          }
        }
      }
    }
  }

  /**
   * Check for variable-time instructions on secret data
   */
  void checkVariableTimeInstructions(Function &F) {
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        auto *BO = dyn_cast<BinaryOperator>(&I);
        if (!BO)
          continue;

        // Check for division and modulo operations
        if (DisableDiv && (BO->getOpcode() == Instruction::UDiv ||
                           BO->getOpcode() == Instruction::SDiv ||
                           BO->getOpcode() == Instruction::URem ||
                           BO->getOpcode() == Instruction::SRem)) {

          // Check if any operand is secret
          if (isSecretValue(BO->getOperand(0)) || isSecretValue(BO->getOperand(1))) {
            std::string OpName;
            switch (BO->getOpcode()) {
              case Instruction::UDiv: OpName = "udiv"; break;
              case Instruction::SDiv: OpName = "sdiv"; break;
              case Instruction::URem: OpName = "urem"; break;
              case Instruction::SRem: OpName = "srem"; break;
              default: OpName = "unknown"; break;
            }

            std::string Msg = "Variable-time instruction: " + OpName + " on secret data " +
                              "(division/modulo has data-dependent timing)";
            Violations.emplace_back(ViolationType::VariableTimeInstruction,
                                    F.getName(), BO, Msg);
          }
        }

        // Check for variable-time shifts (shifts by secret amounts)
        if (BO->getOpcode() == Instruction::Shl ||
            BO->getOpcode() == Instruction::LShr ||
            BO->getOpcode() == Instruction::AShr) {

          // Shifting BY a secret amount is timing-dependent
          if (isSecretValue(BO->getOperand(1))) {
            std::string Msg =
                std::string("Variable-time instruction: shift by secret amount ") +
                "(shift timing may depend on shift count)";
            Violations.emplace_back(ViolationType::VariableTimeInstruction,
                                    F.getName(), BO, Msg);
          }
        }
      }
    }
  }

  /**
   * Check for potential secret leaks through return values
   */
  void checkSecretLeaks(Function &F) {
    // If function is NOT marked as secret, but returns secret values, warn
    if (hasDsmilSecretAttr(F)) {
      // Returning secrets from secret function is OK
      return;
    }

    for (BasicBlock &BB : F) {
      auto *RI = dyn_cast<ReturnInst>(BB.getTerminator());
      if (!RI)
        continue;

      Value *RetVal = RI->getReturnValue();
      if (RetVal && isSecretValue(RetVal)) {
        std::string Msg = "Potential secret leak: non-secret function returning secret value";
        Violations.emplace_back(ViolationType::SecretLeak,
                                F.getName(), RI, Msg);
      }
    }
  }

  /**
   * Analyze function for constant-time violations
   */
  bool analyzeFunction(Function &F) {
    // Skip declarations
    if (F.isDeclaration())
      return false;

    LLVM_DEBUG(dbgs() << "Analyzing constant-time properties of function: " << F.getName() << "\n");

    // Initialize secret tainting
    initializeSecretTainting(F);

    // Propagate taints through SSA
    propagateSecretTaint(F);

    // Run all checks
    checkSecretBranches(F);
    checkSecretMemoryAccess(F);
    checkVariableTimeInstructions(F);
    checkSecretLeaks(F);

    return !Violations.empty();
  }

  /**
   * Print violations report
   */
  void printViolations(Module &M) {
    if (Violations.empty()) {
      outs() << "[DSMIL Constant-Time] No violations found in module " << M.getName() << "\n";
      return;
    }

    errs() << "[DSMIL Constant-Time] Found " << Violations.size() << " violations:\n\n";

    for (const auto &V : Violations) {
      const char *TypeStr = "";
      switch (V.Type) {
        case ViolationType::SecretDependentBranch:
          TypeStr = "SECRET_BRANCH";
          break;
        case ViolationType::SecretDependentMemory:
          TypeStr = "SECRET_MEMORY";
          break;
        case ViolationType::VariableTimeInstruction:
          TypeStr = "VARIABLE_TIME";
          break;
        case ViolationType::SecretLeak:
          TypeStr = "SECRET_LEAK";
          break;
      }

      errs() << "  [" << TypeStr << "] " << V.FunctionName;
      if (V.LineNumber > 0) {
        errs() << ":" << V.LineNumber;
      }
      errs() << "\n";
      errs() << "    " << V.Message << "\n";
      if (!V.InstructionDesc.empty()) {
        errs() << "    Instruction: " << V.InstructionDesc << "\n";
      }
      errs() << "\n";
    }

    // Fail the build in strict mode or if there are critical violations
    if (StrictMode) {
      report_fatal_error("Constant-time violations detected in strict mode");
    }
  }

  /**
   * Add metadata to IR indicating constant-time verification status
   */
  void addVerificationMetadata(Function &F, bool HasViolations) {
    LLVMContext &Ctx = F.getContext();

    if (!HasViolations && hasDsmilSecretAttr(F)) {
      // Mark function as constant-time verified
      MDNode *MD = MDNode::get(Ctx, MDString::get(Ctx, "verified"));
      F.setMetadata("dsmil.ct_verified", MD);
    }
  }

public:
  DsmilConstantTimePass()
      : Enabled(EnableCTCheck),
        StrictMode(CTCheckStrict),
        DisableDiv(CTCheckDisableDiv),
        OutputPath(CTCheckOutput) {}

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM) {
    if (!Enabled) {
      return PreservedAnalyses::all();
    }

    LLVM_DEBUG(dbgs() << "[DSMIL Constant-Time] Analyzing module: " << M.getName() << "\n");

    Violations.clear();

    // Analyze each function
    for (Function &F : M) {
      bool HasViolations = analyzeFunction(F);
      addVerificationMetadata(F, HasViolations);
    }

    // Print violations report
    printViolations(M);

    // IR is not modified, only metadata added
    PreservedAnalyses PA;
    PA.preserve<DominatorTreeAnalysis>();
    PA.preserve<PostDominatorTreeAnalysis>();
    return PA;
  }

  static bool isRequired() { return true; }
};

} // anonymous namespace

// Pass registration (for new pass manager)
extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "DsmilConstantTimePass", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &MPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "dsmil-ct-check") {
                    MPM.addPass(DsmilConstantTimePass());
                    return true;
                  }
                  return false;
                });
          }};
}
