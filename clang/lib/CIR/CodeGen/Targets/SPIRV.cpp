//===---- SPIRV.cpp - SPIR-V-specific CIR CodeGen -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides SPIR/SPIRV-specific CIR CodeGen logic for function attributes.
//
//===----------------------------------------------------------------------===//

#include "../CIRGenModule.h"
#include "../TargetInfo.h"

#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"

using namespace clang;
using namespace clang::CIRGen;

namespace {

class CommonSPIRABIInfo : public ABIInfo {
public:
  CommonSPIRABIInfo(CIRGenTypes &cgt) : ABIInfo(cgt) {}
};

class CommonSPIRTargetCIRGenInfo : public TargetCIRGenInfo {
public:
  CommonSPIRTargetCIRGenInfo(CIRGenTypes &cgt)
      : TargetCIRGenInfo(std::make_unique<CommonSPIRABIInfo>(cgt)) {}

  mlir::ptr::MemorySpaceAttrInterface
  getCIRAllocaAddressSpace() const override {
    return cir::LangAddressSpaceAttr::get(
        &getABIInfo().cgt.getMLIRContext(),
        cir::LangAddressSpace::OffloadPrivate);
  }

  void setTargetAttributes(const clang::Decl *decl, mlir::Operation *global,
                           CIRGenModule &cgm) const override {
    auto func = mlir::dyn_cast<cir::FuncOp>(global);
    if (!func || func.isDeclaration())
      return;

    const auto *fd = dyn_cast_or_null<FunctionDecl>(decl);
    if (!fd)
      return;

    if (!cgm.getLangOpts().HIP || !cgm.getTriple().isSPIRV() ||
        cgm.getTriple().getVendor() != llvm::Triple::AMD)
      return;

    if (!fd->hasAttr<CUDAGlobalAttr>())
      return;

    unsigned n = cgm.getLangOpts().GPUMaxThreadsPerBlock;
    if (const auto *flatWGS = fd->getAttr<AMDGPUFlatWorkGroupSizeAttr>()) {
      n = flatWGS->getMax()
              ->EvaluateKnownConstInt(cgm.getASTContext())
              .getExtValue();
    } else if (const auto *lb = fd->getAttr<CUDALaunchBoundsAttr>()) {
      if (uint64_t maxThreads = lb->getMaxThreads()
                                    ->EvaluateKnownConstInt(cgm.getASTContext())
                                    .getExtValue())
        n = maxThreads;
    }

    // Only x carries the flat WG size, reverse translated for AMDGPU targets.
    func->setAttr(
        cir::CIRDialect::getMaxWorkGroupSizeAttrName(),
        cir::MaxWorkGroupSizeAttr::get(func.getContext(), n, /*y=*/1, /*z=*/1));
  }

  cir::CallingConv getDeviceKernelCallingConv() const override {
    return cir::CallingConv::SpirKernel;
  }

  void setCUDAKernelCallingConvention(const FunctionType *&ft) const override {
    // Convert HIP kernels to SPIR-V kernels.
    if (getABIInfo().cgt.getASTContext().getLangOpts().HIP)
      ft = getABIInfo().cgt.getASTContext().adjustFunctionType(
          ft, ft->getExtInfo().withCallingConv(CC_DeviceKernel));
  }
};

} // namespace

std::unique_ptr<TargetCIRGenInfo>
clang::CIRGen::createCommonSPIRTargetCIRGenInfo(CIRGenTypes &cgt) {
  return std::make_unique<CommonSPIRTargetCIRGenInfo>(cgt);
}
