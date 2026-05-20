//===- raiser.cpp - Hotswap MC -> LLVM IR raiser scaffolding --------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "raiser.h"

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/TargetParser/AMDGPUTargetParser.h"
#include "llvm/TargetParser/Triple.h"

namespace COMGR::hotswap {

namespace {

constexpr llvm::StringLiteral AMDGPUTriple = "amdgcn-amd-amdhsa";

// Reject obviously-bad inputs before constructing IR. Mirrors the
// preconditions the full pipeline enforces in subsequent commits.
//
// Ideally we would reuse `COMGR::parseTargetIdentifier`, but that helper
// currently lives behind the comgr-metadata layer in `src/comgr.cpp` and
// is not reachable from the hotswap subproject. As a stop-gap, validate
// the AMDGPU processor name through `llvm::AMDGPU::parseArchAMDGCN`.
RaiseFailure validateInputs(llvm::StringRef SourceISA,
                            llvm::StringRef KernelName,
                            const KernelMeta &Meta) {
  RaiseFailure F;
  if (SourceISA.empty()) {
    F.Reason = RaiseFailureReason::BadInput;
    F.Detail = "source ISA string is empty";
    return F;
  }
  // The disassembler-facing identifier is `<arch>-<vendor>-<os>-<env>-<gfx>`;
  // `parseArchAMDGCN` inspects the trailing component.
  llvm::StringRef GfxName = SourceISA.rsplit('-').second;
  if (GfxName.empty()) {
    GfxName = SourceISA;
  }
  if (llvm::AMDGPU::parseArchAMDGCN(GfxName) == llvm::AMDGPU::GK_NONE) {
    F.Reason = RaiseFailureReason::BadInput;
    F.Detail =
        ("source ISA '" + SourceISA + "' does not name an AMDGPU GPU").str();
    return F;
  }
  if (KernelName.empty()) {
    F.Reason = RaiseFailureReason::BadInput;
    F.Detail = "kernel name is empty";
    return F;
  }
  if (!Meta.HasKernelDescriptor) {
    F.Reason = RaiseFailureReason::BadInput;
    F.Detail = ("kernel '" + KernelName + "' has no parsed kernel descriptor")
                   .str();
    return F;
  }
  return F;
}

} // namespace

RaiseResult raiseToIR(llvm::StringRef SourceISA,
                      llvm::StringRef KernelName,
                      const KernelMeta &Meta) {
  using namespace llvm;

  RaiseResult Result;
  Result.Failure = validateInputs(SourceISA, KernelName, Meta);
  if (Result.Failure.hasFailed()) {
    return Result;
  }

  Result.Ctx = std::make_unique<LLVMContext>();
  LLVMContext &C = *Result.Ctx;
  Result.Module = std::make_unique<Module>("transpiler_module", C);
  Module &M = *Result.Module;
  M.setTargetTriple(Triple(AMDGPUTriple));

  FunctionType *FuncTy =
      FunctionType::get(Type::getVoidTy(C), /*isVarArg=*/false);
  Function *F =
      Function::Create(FuncTy, GlobalValue::ExternalLinkage, KernelName, &M);
  F->setCallingConv(CallingConv::AMDGPU_KERNEL);

  BasicBlock *Entry = BasicBlock::Create(C, "entry", F);
  IRBuilder<> B(Entry);
  B.CreateRetVoid();

  Result.Success = true;
  return Result;
}

} // namespace COMGR::hotswap
