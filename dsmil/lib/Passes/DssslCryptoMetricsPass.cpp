/**
 * @file DssslCryptoMetricsPass.cpp
 * @brief DSSSL Crypto Metrics Instrumentation Pass
 *
 * Instruments crypto functions with branch, load/store, and timing metrics
 * for side-channel detection.
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include <map>
#include <string>

#define DEBUG_TYPE "dsssl-crypto-metrics"

using namespace llvm;

// Command-line options
static cl::opt<bool> EnableCryptoMetrics(
    "dsssl-crypto-metrics",
    cl::desc("Enable DSSSL crypto metrics instrumentation"),
    cl::init(false));

static cl::opt<bool> EnableTiming(
    "dsssl-crypto-timing",
    cl::desc("Enable timing measurements (rdtsc/clock_gettime)"),
    cl::init(false));

namespace {

/**
 * Crypto Metrics Instrumentation Pass
 */
class DssslCryptoMetricsPass : public PassInfoMixin<DssslCryptoMetricsPass> {
private:
    Module *M;

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
     * Get or create crypto metric begin function
     */
    Function* getCryptoBeginFunction() {
        FunctionType *FTy = FunctionType::get(
            Type::getVoidTy(M->getContext()),
            {PointerType::getInt8PtrTy(M->getContext())},  // op_name
            false);

        Function *F = M->getFunction("dsssl_crypto_metric_begin");
        if (!F) {
            F = Function::Create(FTy, Function::ExternalLinkage,
                                "dsssl_crypto_metric_begin", *M);
            F->setCallingConv(CallingConv::C);
        }
        return F;
    }

    /**
     * Get or create crypto metric end function
     */
    Function* getCryptoEndFunction() {
        FunctionType *FTy = FunctionType::get(
            Type::getVoidTy(M->getContext()),
            {PointerType::getInt8PtrTy(M->getContext())},  // op_name
            false);

        Function *F = M->getFunction("dsssl_crypto_metric_end");
        if (!F) {
            F = Function::Create(FTy, Function::ExternalLinkage,
                                "dsssl_crypto_metric_end", *M);
            F->setCallingConv(CallingConv::C);
        }
        return F;
    }

    /**
     * Count branches in function
     */
    uint32_t countBranches(Function &F) {
        uint32_t count = 0;
        for (BasicBlock &BB : F) {
            if (isa<BranchInst>(BB.getTerminator()) || isa<SwitchInst>(BB.getTerminator())) {
                count++;
            }
        }
        return count;
    }

    /**
     * Count loads/stores in function
     */
    void countMemoryOps(Function &F, uint32_t &loads, uint32_t &stores) {
        loads = 0;
        stores = 0;
        for (BasicBlock &BB : F) {
            for (Instruction &I : BB) {
                if (isa<LoadInst>(&I)) loads++;
                if (isa<StoreInst>(&I)) stores++;
            }
        }
    }

    /**
     * Instrument crypto function
     */
    void instrumentCryptoFunction(Function &F, const std::string &op_name) {
        if (F.isDeclaration()) return;

        LLVMContext &Ctx = M->getContext();
        Function *BeginFn = getCryptoBeginFunction();
        Function *EndFn = getCryptoEndFunction();

        // Create string constant for operation name
        Constant *OpNameStr = ConstantDataArray::getString(Ctx, op_name, true);
        GlobalVariable *OpNameGV = new GlobalVariable(
            *M, OpNameStr->getType(), true, GlobalValue::PrivateLinkage,
            OpNameStr, "crypto_op_name");

        // Instrument function entry
        BasicBlock &EntryBB = F.getEntryBlock();
        IRBuilder<> EntryBuilder(&EntryBB, EntryBB.begin());
        Value *OpNamePtr = EntryBuilder.CreateConstGEP2_32(
            OpNameStr->getType(), OpNameGV, 0, 0);
        EntryBuilder.CreateCall(BeginFn, {OpNamePtr});

        // Instrument function exits
        for (BasicBlock &BB : F) {
            Instruction *Term = BB.getTerminator();
            if (isa<ReturnInst>(Term) || isa<ResumeInst>(Term)) {
                IRBuilder<> ExitBuilder(Term);
                ExitBuilder.CreateCall(EndFn, {OpNamePtr});
            }
        }

        // Note: Full implementation would track branches/loads/stores dynamically
        // This is a simplified version that instruments begin/end calls
    }

public:
    PreservedAnalyses run(Module &Mod, ModuleAnalysisManager &MAM) {
        if (!EnableCryptoMetrics) {
            return PreservedAnalyses::all();
        }

        M = &Mod;

        outs() << "[DSSSL Crypto Metrics] Instrumenting module: " << Mod.getName() << "\n";

        int crypto_funcs = 0;
        for (Function &F : Mod) {
            if (F.isDeclaration()) continue;

            std::string op_name = extractAnnotationParam(F, "dsssl.crypto");
            if (!op_name.empty()) {
                instrumentCryptoFunction(F, op_name);
                crypto_funcs++;
            }
        }

        outs() << "  Crypto functions instrumented: " << crypto_funcs << "\n";

        return PreservedAnalyses::none();  // We modified the IR
    }
};

} // namespace

// Pass registration
extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
    return {
        LLVM_PLUGIN_API_VERSION, "DssslCryptoMetricsPass", LLVM_VERSION_STRING,
        [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &MPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                    if (Name == "dsssl-crypto-metrics") {
                        MPM.addPass(DssslCryptoMetricsPass());
                        return true;
                    }
                    return false;
                });
        }
    };
}
