/**
 * @file DsmilStealthPass.cpp
 * @brief DSLLVM Stealth Mode Transformation Pass (v1.4 - Feature 2.1)
 *
 * This pass implements "Operational Stealth" transformations for binaries
 * deployed in hostile network environments. It reduces detectability through:
 * - Telemetry reduction (strip non-critical logging)
 * - Constant-rate execution (timing normalization)
 * - Jitter suppression (predictable timing)
 * - Network fingerprint reduction (batched/delayed I/O)
 *
 * Integrates with Layer 5/8 AI to model detectability vs debugging trade-offs.
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include <map>
#include <set>
#include <string>
#include <vector>

#define DEBUG_TYPE "dsmil-stealth"

using namespace llvm;

// Command-line options
static cl::opt<std::string> StealthMode(
    "dsmil-stealth-mode",
    cl::desc("Stealth transformation mode (off, minimal, standard, aggressive)"),
    cl::init("off"));

static cl::opt<bool> StripTelemetry(
    "dsmil-stealth-strip-telemetry",
    cl::desc("Strip non-critical telemetry calls in stealth mode"),
    cl::init(true));

static cl::opt<bool> ConstantRateExecution(
    "dsmil-stealth-constant-rate",
    cl::desc("Enable constant-rate execution transformations"),
    cl::init(false));

static cl::opt<bool> JitterSuppression(
    "dsmil-stealth-jitter-suppress",
    cl::desc("Enable jitter suppression optimizations"),
    cl::init(false));

static cl::opt<bool> NetworkFingerprint(
    "dsmil-stealth-network-reduce",
    cl::desc("Enable network fingerprint reduction"),
    cl::init(false));

static cl::opt<unsigned> ConstantRateTargetMs(
    "dsmil-stealth-rate-target-ms",
    cl::desc("Target execution time in milliseconds for constant-rate functions"),
    cl::init(100));

static cl::opt<bool> PreserveSafetyCritical(
    "dsmil-stealth-preserve-safety",
    cl::desc("Always preserve safety-critical telemetry even in stealth mode"),
    cl::init(true));

namespace {

/**
 * Stealth level enumeration
 */
enum StealthLevel {
  STEALTH_OFF = 0,       // No stealth transformations
  STEALTH_MINIMAL = 1,   // Basic telemetry reduction only
  STEALTH_STANDARD = 2,  // Moderate stealth (timing + telemetry)
  STEALTH_AGGRESSIVE = 3 // Maximum stealth (all transformations)
};

/**
 * Telemetry call classification
 */
enum TelemetryClass {
  TELEMETRY_CRITICAL,     // Must keep (safety/mission critical)
  TELEMETRY_STANDARD,     // Standard telemetry
  TELEMETRY_VERBOSE,      // Verbose/debug telemetry
  TELEMETRY_PERFORMANCE   // Performance metrics
};

/**
 * Stealth Transformation Pass
 */
class DsmilStealthPass : public PassInfoMixin<DsmilStealthPass> {
private:
  std::string Mode;
  bool StripTelem;
  bool ConstantRate;
  bool JitterSuppress;
  bool NetworkReduce;
  unsigned RateTargetMs;
  bool PreserveSafety;

  // Statistics
  unsigned FunctionsTransformed = 0;
  unsigned TelemetryCallsStripped = 0;
  unsigned ConstantRateFunctionsAdded = 0;
  unsigned NetworkCallsModified = 0;

  /**
   * Parse stealth level from attribute or CLI
   */
  StealthLevel getStealthLevel(Function &F) {
    // Check function attributes first
    if (F.hasFnAttribute("dsmil_low_signature")) {
      Attribute Attr = F.getFnAttribute("dsmil_low_signature");
      StringRef Level = Attr.getValueAsString();

      if (Level == "minimal")
        return STEALTH_MINIMAL;
      else if (Level == "standard")
        return STEALTH_STANDARD;
      else if (Level == "aggressive")
        return STEALTH_AGGRESSIVE;
    }

    // Fall back to CLI option
    if (Mode == "minimal")
      return STEALTH_MINIMAL;
    else if (Mode == "standard")
      return STEALTH_STANDARD;
    else if (Mode == "aggressive")
      return STEALTH_AGGRESSIVE;

    return STEALTH_OFF;
  }

  /**
   * Check if function is safety-critical or mission-critical
   */
  bool isCriticalFunction(Function &F) {
    return F.hasFnAttribute("dsmil_safety_critical") ||
           F.hasFnAttribute("dsmil_mission_critical");
  }

  /**
   * Classify telemetry call
   */
  TelemetryClass classifyTelemetryCall(CallInst *CI) {
    Function *Callee = CI->getCalledFunction();
    if (!Callee)
      return TELEMETRY_STANDARD;

    StringRef Name = Callee->getName();

    // Critical telemetry (always keep)
    if (Name.contains("dsmil_forensic") ||
        Name.contains("dsmil_security_event") ||
        Name.contains("critical"))
      return TELEMETRY_CRITICAL;

    // Performance metrics
    if (Name.contains("dsmil_perf") ||
        Name.contains("dsmil_counter"))
      return TELEMETRY_PERFORMANCE;

    // Verbose/debug
    if (Name.contains("debug") ||
        Name.contains("verbose") ||
        Name.contains("trace"))
      return TELEMETRY_VERBOSE;

    return TELEMETRY_STANDARD;
  }

  /**
   * Strip non-critical telemetry calls
   */
  bool stripTelemetryCalls(Function &F, StealthLevel Level) {
    if (!StripTelem || Level == STEALTH_OFF)
      return false;

    std::vector<CallInst *> ToRemove;
    bool Modified = false;

    for (auto &BB : F) {
      for (auto &I : BB) {
        if (auto *CI = dyn_cast<CallInst>(&I)) {
          Function *Callee = CI->getCalledFunction();
          if (!Callee)
            continue;

          StringRef Name = Callee->getName();

          // Skip if not a telemetry call
          if (!Name.starts_with("dsmil_counter") &&
              !Name.starts_with("dsmil_event") &&
              !Name.starts_with("dsmil_perf") &&
              !Name.starts_with("dsmil_trace"))
            continue;

          TelemetryClass Class = classifyTelemetryCall(CI);

          // Always keep critical telemetry
          if (Class == TELEMETRY_CRITICAL)
            continue;

          // Keep safety-critical telemetry if preserving
          if (PreserveSafety && isCriticalFunction(F))
            continue;

          // Strip based on stealth level
          bool ShouldStrip = false;
          switch (Level) {
            case STEALTH_MINIMAL:
              // Only strip verbose/debug
              ShouldStrip = (Class == TELEMETRY_VERBOSE);
              break;

            case STEALTH_STANDARD:
              // Strip verbose and some standard telemetry
              ShouldStrip = (Class == TELEMETRY_VERBOSE ||
                           Class == TELEMETRY_PERFORMANCE);
              break;

            case STEALTH_AGGRESSIVE:
              // Strip all non-critical
              ShouldStrip = (Class != TELEMETRY_CRITICAL);
              break;

            default:
              break;
          }

          if (ShouldStrip) {
            ToRemove.push_back(CI);
            TelemetryCallsStripped++;
            Modified = true;
          }
        }
      }
    }

    // Remove marked calls
    for (auto *CI : ToRemove) {
      CI->eraseFromParent();
    }

    return Modified;
  }

  /**
   * Add constant-rate execution padding
   */
  bool addConstantRatePadding(Function &F, StealthLevel Level) {
    if (!ConstantRate && !F.hasFnAttribute("dsmil_constant_rate"))
      return false;

    if (Level < STEALTH_STANDARD)
      return false;

    // Find all return instructions
    std::vector<ReturnInst *> Returns;
    for (auto &BB : F) {
      if (auto *RI = dyn_cast<ReturnInst>(BB.getTerminator())) {
        Returns.push_back(RI);
      }
    }

    if (Returns.empty())
      return false;

    // Insert timing logic at function entry
    BasicBlock &Entry = F.getEntryBlock();
    IRBuilder<> EntryBuilder(&Entry, Entry.getFirstInsertionPt());

    // Get current timestamp (nanoseconds)
    Module *M = F.getParent();
    LLVMContext &Ctx = M->getContext();

    FunctionCallee GetTimeFunc = M->getOrInsertFunction(
        "dsmil_get_timestamp_ns",
        Type::getInt64Ty(Ctx));

    Value *StartTime = EntryBuilder.CreateCall(GetTimeFunc);

    // Store start time in a local variable
    AllocaInst *StartTimeAlloca = EntryBuilder.CreateAlloca(
        Type::getInt64Ty(Ctx), nullptr, "stealth_start_time");
    EntryBuilder.CreateStore(StartTime, StartTimeAlloca);

    // Insert delay logic before each return
    uint64_t TargetNs = RateTargetMs * 1000000ULL; // Convert ms to ns

    for (auto *RI : Returns) {
      IRBuilder<> RetBuilder(RI);

      // Load start time
      Value *Start = RetBuilder.CreateLoad(Type::getInt64Ty(Ctx), StartTimeAlloca);

      // Get current time
      Value *CurrentTime = RetBuilder.CreateCall(GetTimeFunc);

      // Calculate elapsed time
      Value *Elapsed = RetBuilder.CreateSub(CurrentTime, Start);

      // Calculate required delay: max(0, TargetNs - Elapsed)
      Value *TargetNsVal = ConstantInt::get(Type::getInt64Ty(Ctx), TargetNs);
      Value *RequiredDelay = RetBuilder.CreateSub(TargetNsVal, Elapsed);

      // Only delay if positive
      Value *ShouldDelay = RetBuilder.CreateICmpSGT(
          RequiredDelay, ConstantInt::get(Type::getInt64Ty(Ctx), 0));

      // Create conditional delay
      BasicBlock *DelayBB = BasicBlock::Create(Ctx, "stealth_delay", &F);
      BasicBlock *ContBB = BasicBlock::Create(Ctx, "stealth_continue", &F);

      // Replace return with conditional branch
      RetBuilder.CreateCondBr(ShouldDelay, DelayBB, ContBB);
      RI->removeFromParent();

      // Delay block: call sleep function
      IRBuilder<> DelayBuilder(DelayBB);
      FunctionCallee DelayFunc = M->getOrInsertFunction(
          "dsmil_nanosleep",
          Type::getVoidTy(Ctx),
          Type::getInt64Ty(Ctx));
      DelayBuilder.CreateCall(DelayFunc, {RequiredDelay});
      DelayBuilder.CreateBr(ContBB);

      // Continue block: emit return
      IRBuilder<> ContBuilder(ContBB);
      ContBuilder.CreateRetVoid();
    }

    ConstantRateFunctionsAdded++;
    return true;
  }

  /**
   * Apply jitter suppression optimizations
   */
  bool applyJitterSuppression(Function &F, StealthLevel Level) {
    if (!JitterSuppress && !F.hasFnAttribute("dsmil_jitter_suppress"))
      return false;

    if (Level < STEALTH_STANDARD)
      return false;

    // Add function attributes to hint optimizer
    F.addFnAttr("no-jump-tables"); // Avoid jump table timing variance
    F.addFnAttr("prefer-vector-width", "256"); // Consistent vector width

    // Disable some optimizations that introduce timing variance
    if (Level == STEALTH_AGGRESSIVE) {
      F.addFnAttr(Attribute::OptimizeForSize); // More predictable code size
    }

    return true;
  }

  /**
   * Transform network calls for fingerprint reduction
   */
  bool transformNetworkCalls(Function &F, StealthLevel Level) {
    if (!NetworkReduce && !F.hasFnAttribute("dsmil_network_stealth"))
      return false;

    if (Level < STEALTH_MINIMAL)
      return false;

    bool Modified = false;
    Module *M = F.getParent();
    LLVMContext &Ctx = M->getContext();

    // Create batching/delay wrapper for network calls
    FunctionCallee NetworkWrapperFunc = M->getOrInsertFunction(
        "dsmil_network_stealth_wrapper",
        Type::getVoidTy(Ctx),
        PointerType::get(Type::getInt8Ty(Ctx), 0), // data
        Type::getInt64Ty(Ctx)    // length
    );

    std::vector<CallInst *> ToWrap;

    for (auto &BB : F) {
      for (auto &I : BB) {
        if (auto *CI = dyn_cast<CallInst>(&I)) {
          Function *Callee = CI->getCalledFunction();
          if (!Callee)
            continue;

          StringRef Name = Callee->getName();

          // Identify network calls (send, write, sendto, sendmsg, etc.)
          if (Name == "send" || Name == "write" || Name == "sendto" ||
              Name == "sendmsg" || Name.contains("network_send")) {

            // For aggressive mode, wrap network calls
            if (Level == STEALTH_AGGRESSIVE) {
              ToWrap.push_back(CI);
            }
          }
        }
      }
    }

    // Actually wrap the network calls
    for (auto *CI : ToWrap) {
      IRBuilder<> Builder(CI);

      // Extract data pointer and length from original call
      // For send(sockfd, buf, len, flags), we want buf (arg 1) and len (arg 2)
      // For write(fd, buf, count), we want buf (arg 1) and count (arg 2)
      Value *DataPtr = nullptr;
      Value *DataLen = nullptr;

      unsigned NumArgs = CI->arg_size();
      if (NumArgs >= 3) {
        // Typical send(sockfd, buf, len, ...) or write(fd, buf, count)
        DataPtr = CI->getArgOperand(1);  // buf/data pointer
        DataLen = CI->getArgOperand(2);  // len/count

        // Cast data pointer to i8* for wrapper
        auto *I8PtrTy = PointerType::get(Type::getInt8Ty(Ctx), 0);
        if (DataPtr->getType() != I8PtrTy) {
          DataPtr = Builder.CreateBitCast(DataPtr, I8PtrTy);
        }

        // Ensure length is i64
        if (DataLen->getType() != Type::getInt64Ty(Ctx)) {
          DataLen = Builder.CreateZExtOrTrunc(DataLen, Type::getInt64Ty(Ctx));
        }

        // Insert stealth wrapper call BEFORE the original send
        Builder.CreateCall(NetworkWrapperFunc, {DataPtr, DataLen});

        NetworkCallsModified++;
        Modified = true;

        LLVM_DEBUG(dbgs() << "  [Stealth] Wrapped network call: "
                          << CI->getCalledFunction()->getName() << "\n");
      }
    }

    return Modified;
  }

  /**
   * Add stealth metadata to function
   */
  void addStealthMetadata(Function &F, StealthLevel Level) {
    Module *M = F.getParent();
    LLVMContext &Ctx = M->getContext();

    // Create metadata node
    SmallVector<Metadata *, 4> MDVals;
    MDVals.push_back(MDString::get(Ctx, "dsmil.stealth.level"));

    const char *LevelStr = "off";
    switch (Level) {
      case STEALTH_MINIMAL: LevelStr = "minimal"; break;
      case STEALTH_STANDARD: LevelStr = "standard"; break;
      case STEALTH_AGGRESSIVE: LevelStr = "aggressive"; break;
      default: break;
    }
    MDVals.push_back(MDString::get(Ctx, LevelStr));

    MDNode *MD = MDNode::get(Ctx, MDVals);
    F.setMetadata("dsmil.stealth", MD);
  }

public:
  DsmilStealthPass()
    : Mode(StealthMode.getValue()),
      StripTelem(StripTelemetry.getValue()),
      ConstantRate(ConstantRateExecution.getValue()),
      JitterSuppress(JitterSuppression.getValue()),
      NetworkReduce(NetworkFingerprint.getValue()),
      RateTargetMs(ConstantRateTargetMs.getValue()),
      PreserveSafety(PreserveSafetyCritical.getValue()) {}

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM) {
    bool Modified = false;

    LLVM_DEBUG(dbgs() << "[DSMIL Stealth] Processing module: "
                      << M.getName() << "\n");
    LLVM_DEBUG(dbgs() << "[DSMIL Stealth] Mode: " << Mode << "\n");

    for (auto &F : M) {
      if (F.isDeclaration())
        continue;

      StealthLevel Level = getStealthLevel(F);

      if (Level == STEALTH_OFF)
        continue;

      LLVM_DEBUG(dbgs() << "[DSMIL Stealth] Transforming function: "
                        << F.getName() << " (level: " << (int)Level << ")\n");

      bool FuncModified = false;

      // Apply transformations
      FuncModified |= stripTelemetryCalls(F, Level);
      FuncModified |= addConstantRatePadding(F, Level);
      FuncModified |= applyJitterSuppression(F, Level);
      FuncModified |= transformNetworkCalls(F, Level);

      if (FuncModified) {
        addStealthMetadata(F, Level);
        FunctionsTransformed++;
        Modified = true;
      }
    }

    // Print statistics
    if (Modified) {
      errs() << "[DSMIL Stealth] Transformation Summary:\n";
      errs() << "  Functions transformed: " << FunctionsTransformed << "\n";
      errs() << "  Telemetry calls stripped: " << TelemetryCallsStripped << "\n";
      errs() << "  Constant-rate functions: " << ConstantRateFunctionsAdded << "\n";
      errs() << "  Network calls modified: " << NetworkCallsModified << "\n";
    }

    return Modified ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }

  static bool isRequired() { return true; }
};

} // end anonymous namespace

// Register the pass
extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {
    LLVM_PLUGIN_API_VERSION, "DsmilStealthPass", LLVM_VERSION_STRING,
    [](PassBuilder &PB) {
      PB.registerPipelineParsingCallback(
        [](StringRef Name, ModulePassManager &MPM,
           ArrayRef<PassBuilder::PipelineElement>) {
          if (Name == "dsmil-stealth") {
            MPM.addPass(DsmilStealthPass());
            return true;
          }
          return false;
        });
    }
  };
}
