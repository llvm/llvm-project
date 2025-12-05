/**
 * @file DssslCoveragePass.cpp
 * @brief DSSSL Coverage & State Machine Instrumentation Pass
 *
 * Instruments functions with coverage counters and state machine transition
 * tracking for fuzzing.
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include <map>
#include <set>
#include <string>

#define DEBUG_TYPE "dsssl-coverage"

using namespace llvm;

// Command-line options
static cl::opt<bool> EnableCoverage(
    "dsssl-coverage",
    cl::desc("Enable DSSSL coverage instrumentation"),
    cl::init(false));

static cl::opt<bool> EnableStateMachine(
    "dsssl-state-machine",
    cl::desc("Enable DSSSL state machine instrumentation"),
    cl::init(false));

namespace {

/**
 * Coverage Instrumentation Pass
 */
class DssslCoveragePass : public PassInfoMixin<DssslCoveragePass> {
private:
    Module *M;
    uint32_t NextSiteID;
    std::map<std::string, uint16_t> StateMachineIDs;
    uint16_t NextSMID;

    /**
     * Check if function has annotation
     */
    bool hasAnnotation(Function &F, StringRef AttrName) {
        if (MDNode *MD = F.getMetadata("llvm.ptr.annotation")) {
            for (unsigned i = 0; i < MD->getNumOperands(); i++) {
                if (MDString *Str = dyn_cast<MDString>(MD->getOperand(i))) {
                    if (Str->getString().startswith(AttrName)) {
                        return true;
                    }
                }
            }
        }
        
        if (F.hasFnAttribute("annotate")) {
            Attribute Attr = F.getFnAttribute("annotate");
            if (Attr.isStringAttribute()) {
                StringRef Value = Attr.getValueAsString();
                return Value.startswith(AttrName);
            }
        }
        
        return false;
    }

    /**
     * Extract annotation parameter value
     */
    std::string extractAnnotationParam(Function &F, StringRef AttrName) {
        if (MDNode *MD = F.getMetadata("llvm.ptr.annotation")) {
            for (unsigned i = 0; i < MD->getNumOperands(); i++) {
                if (MDString *Str = dyn_cast<MDString>(MD->getOperand(i))) {
                    StringRef Value = Str->getString();
                    if (Value.startswith(AttrName + "=")) {
                        return Value.substr(AttrName.size() + 1).str();
                    }
                }
            }
        }
        
        if (F.hasFnAttribute("annotate")) {
            Attribute Attr = F.getFnAttribute("annotate");
            if (Attr.isStringAttribute()) {
                StringRef Value = Attr.getValueAsString();
                if (Value.startswith(AttrName + "=")) {
                    return Value.substr(AttrName.size() + 1).str();
                }
            }
        }
        
        return "";
    }

    /**
     * Get or create coverage hit function
     */
    Function* getCoverageHitFunction() {
        FunctionType *FTy = FunctionType::get(
            Type::getVoidTy(M->getContext()),
            {Type::getInt32Ty(M->getContext())},  // site_id
            false);

        Function *F = M->getFunction("dsssl_cov_hit");
        if (!F) {
            F = Function::Create(FTy, Function::ExternalLinkage, "dsssl_cov_hit", *M);
            F->setCallingConv(CallingConv::C);
        }
        return F;
    }

    /**
     * Get or create state transition function
     */
    Function* getStateTransitionFunction() {
        FunctionType *FTy = FunctionType::get(
            Type::getVoidTy(M->getContext()),
            {Type::getInt16Ty(M->getContext()),  // sm_id
             Type::getInt16Ty(M->getContext()),  // state_from
             Type::getInt16Ty(M->getContext())}, // state_to
            false);

        Function *F = M->getFunction("dsssl_state_transition");
        if (!F) {
            F = Function::Create(FTy, Function::ExternalLinkage,
                                 "dsssl_state_transition", *M);
            F->setCallingConv(CallingConv::C);
        }
        return F;
    }

    /**
     * Get state machine ID for name
     */
    uint16_t getStateMachineID(const std::string &name) {
        if (StateMachineIDs.count(name) == 0) {
            StateMachineIDs[name] = NextSMID++;
        }
        return StateMachineIDs[name];
    }

    /**
     * Instrument function entry for coverage
     */
    void instrumentCoverage(Function &F) {
        if (F.isDeclaration()) return;

        BasicBlock &EntryBB = F.getEntryBlock();
        IRBuilder<> Builder(&EntryBB, EntryBB.begin());

        uint32_t site_id = NextSiteID++;
        Function *CovFn = getCoverageHitFunction();
        Value *SiteIDVal = ConstantInt::get(Type::getInt32Ty(M->getContext()), site_id);
        Builder.CreateCall(CovFn, {SiteIDVal});
    }

    /**
     * Instrument edges for coverage
     */
    void instrumentEdges(Function &F) {
        if (F.isDeclaration()) return;

        for (BasicBlock &BB : F) {
            Instruction *Term = BB.getTerminator();
            if (BranchInst *BI = dyn_cast<BranchInst>(Term)) {
                if (BI->isConditional()) {
                    IRBuilder<> Builder(BI);
                    uint32_t site_id = NextSiteID++;
                    Function *CovFn = getCoverageHitFunction();
                    Value *SiteIDVal = ConstantInt::get(Type::getInt32Ty(M->getContext()), site_id);
                    Builder.CreateCall(CovFn, {SiteIDVal});
                }
            }
        }
    }

    /**
     * Instrument state machine transitions
     */
    void instrumentStateMachine(Function &F, const std::string &sm_name) {
        if (F.isDeclaration()) return;

        uint16_t sm_id = getStateMachineID(sm_name);
        Function *StateFn = getStateTransitionFunction();

        // Instrument function entry (state transition from 0 = initial)
        BasicBlock &EntryBB = F.getEntryBlock();
        IRBuilder<> Builder(&EntryBB, EntryBB.begin());

        Value *SMIDVal = ConstantInt::get(Type::getInt16Ty(M->getContext()), sm_id);
        Value *StateFromVal = ConstantInt::get(Type::getInt16Ty(M->getContext()), 0);
        Value *StateToVal = ConstantInt::get(Type::getInt16Ty(M->getContext()), 1);
        Builder.CreateCall(StateFn, {SMIDVal, StateFromVal, StateToVal});

        // Note: Full state machine tracking would require more sophisticated
        // analysis to identify actual state transitions. This is a simplified
        // version that marks function entry as a transition.
    }

public:
    PreservedAnalyses run(Module &Mod, ModuleAnalysisManager &MAM) {
        if (!EnableCoverage && !EnableStateMachine) {
            return PreservedAnalyses::all();
        }

        M = &Mod;
        NextSiteID = 1;
        NextSMID = 1;

        outs() << "[DSSSL Coverage] Instrumenting module: " << Mod.getName() << "\n";

        for (Function &F : Mod) {
            if (F.isDeclaration()) continue;

            bool needs_coverage = hasAnnotation(F, "dsssl.coverage") ||
                                  hasAnnotation(F, "dsssl.state_machine") ||
                                  hasAnnotation(F, "dsssl.crypto");

            if (needs_coverage && EnableCoverage) {
                instrumentCoverage(F);
                instrumentEdges(F);
            }

            std::string sm_name = extractAnnotationParam(F, "dsssl.state_machine");
            if (!sm_name.empty() && EnableStateMachine) {
                instrumentStateMachine(F, sm_name);
            }
        }

        outs() << "  Coverage sites: " << NextSiteID << "\n";
        outs() << "  State machines: " << StateMachineIDs.size() << "\n";

        return PreservedAnalyses::none();  // We modified the IR
    }
};

} // namespace

// Pass registration
extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
    return {
        LLVM_PLUGIN_API_VERSION, "DssslCoveragePass", LLVM_VERSION_STRING,
        [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &MPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                    if (Name == "dsssl-coverage") {
                        MPM.addPass(DssslCoveragePass());
                        return true;
                    }
                    return false;
                });
        }
    };
}
