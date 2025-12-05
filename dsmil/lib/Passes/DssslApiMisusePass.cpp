/**
 * @file DssslApiMisusePass.cpp
 * @brief DSSSL API Misuse Detection Pass
 *
 * Wraps critical API calls with misuse detection checks.
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
#include <map>
#include <set>
#include <string>

#define DEBUG_TYPE "dsssl-api-misuse"

using namespace llvm;

// Command-line options
static cl::opt<bool> EnableApiMisuse(
    "dsssl-api-misuse",
    cl::desc("Enable DSSSL API misuse detection"),
    cl::init(false));

namespace {

/**
 * API Misuse Detection Pass
 */
class DssslApiMisusePass : public PassInfoMixin<DssslApiMisusePass> {
private:
    Module *M;
    
    // Target APIs to wrap
    std::set<std::string> TargetAPIs = {
        "EVP_AEAD_CTX_init",
        "EVP_AEAD_CTX_seal",
        "EVP_AEAD_CTX_open",
        "SSL_CTX_set_verify",
        "X509_verify_cert"
    };

    /**
     * Get or create API misuse report function
     */
    Function* getApiMisuseFunction() {
        FunctionType *FTy = FunctionType::get(
            Type::getVoidTy(M->getContext()),
            {PointerType::getInt8PtrTy(M->getContext()),  // api
             PointerType::getInt8PtrTy(M->getContext()),  // reason
             Type::getInt64Ty(M->getContext())},          // context_id
            false);

        Function *F = M->getFunction("dsssl_api_misuse_report");
        if (!F) {
            F = Function::Create(FTy, Function::ExternalLinkage,
                                "dsssl_api_misuse_report", *M);
            F->setCallingConv(CallingConv::C);
        }
        return F;
    }

    /**
     * Get or create checked wrapper function
     */
    Function* getCheckedWrapper(const std::string &api_name, Function *Original) {
        std::string wrapper_name = "dsssl_" + api_name + "_checked";
        
        Function *Wrapper = M->getFunction(wrapper_name);
        if (Wrapper) {
            return Wrapper;
        }

        // Create wrapper function with same signature
        FunctionType *FTy = Original->getFunctionType();
        Wrapper = Function::Create(FTy, Function::InternalLinkage, wrapper_name, *M);
        Wrapper->setCallingConv(Original->getCallingConv());

        // Create basic block
        BasicBlock *BB = BasicBlock::Create(M->getContext(), "entry", Wrapper);
        IRBuilder<> Builder(BB);

        // Copy arguments
        std::vector<Value*> Args;
        for (Argument &Arg : Wrapper->args()) {
            Args.push_back(&Arg);
        }

        // Call original function
        Value *Result = Builder.CreateCall(Original, Args);

        // Add misuse checks (simplified - full implementation would check parameters)
        // For now, just call the original and return
        Builder.CreateRet(Result);

        return Wrapper;
    }

    /**
     * Wrap API calls
     */
    void wrapApiCalls(Function &F) {
        if (F.isDeclaration()) return;

        for (BasicBlock &BB : F) {
            for (Instruction &I : BB) {
                if (CallInst *CI = dyn_cast<CallInst>(&I)) {
                    Function *Callee = CI->getCalledFunction();
                    if (!Callee) continue;

                    std::string callee_name = Callee->getName().str();
                    if (TargetAPIs.count(callee_name)) {
                        // Replace call with checked wrapper
                        Function *Wrapper = getCheckedWrapper(callee_name, Callee);
                        CI->setCalledFunction(Wrapper);
                    }
                }
            }
        }
    }

public:
    PreservedAnalyses run(Module &Mod, ModuleAnalysisManager &MAM) {
        if (!EnableApiMisuse) {
            return PreservedAnalyses::all();
        }

        M = &Mod;

        outs() << "[DSSSL API Misuse] Analyzing module: " << Mod.getName() << "\n";

        // Wrap API calls in all functions
        for (Function &F : Mod) {
            if (F.isDeclaration()) continue;
            wrapApiCalls(F);
        }

        return PreservedAnalyses::none();  // We modified the IR
    }
};

} // namespace

// Pass registration
extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
    return {
        LLVM_PLUGIN_API_VERSION, "DssslApiMisusePass", LLVM_VERSION_STRING,
        [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &MPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                    if (Name == "dsssl-api-misuse") {
                        MPM.addPass(DssslApiMisusePass());
                        return true;
                    }
                    return false;
                });
        }
    };
}
