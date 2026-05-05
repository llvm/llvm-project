//===- raiser.cpp - Hotswap MC -> LLVM IR raiser ---------------------------===//
//
// Produce an empty `llvm::Module` that contains a single kernel function with
// `ret void`. See `raiser.hpp` for the full raise pipeline (ELF ingestion ->
// decode -> per-format handlers -> post-raise analyses).
//
//===----------------------------------------------------------------------===//

#include "raiser.hpp"

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/TargetParser/Triple.h"

namespace transpiler {

RaiseResult raiseToIR(llvm::ArrayRef<uint8_t> /*textBytes*/,
                      llvm::StringRef /*sourceISA*/,
                      llvm::StringRef kernelName,
                      const KernelMeta & /*meta*/,
                      uint64_t /*kernelOffset*/,
                      llvm::StringRef /*compilationTargetISA*/) {
  using namespace llvm;
  RaiseResult result;
  result.ctx = std::make_unique<LLVMContext>();
  LLVMContext &C = *result.ctx;

  result.module = std::make_unique<Module>("transpiler_module", C);
  Module &M = *result.module;
  M.setTargetTriple(Triple("amdgcn-amd-amdhsa"));

  auto *funcTy = FunctionType::get(Type::getVoidTy(C), /*isVarArg=*/false);
  Function *F =
      Function::Create(funcTy, GlobalValue::ExternalLinkage, kernelName, &M);
  F->setCallingConv(CallingConv::AMDGPU_KERNEL);

  BasicBlock *entry = BasicBlock::Create(C, "entry", F);
  IRBuilder<> B(entry);
  B.CreateRetVoid();

  result.success = true;
  return result;
}

} // namespace transpiler
