//===---- NVPTX.cpp - NVPTX-specific CIR CodeGen --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides NVPTX-specific CIR CodeGen logic.
//
//===----------------------------------------------------------------------===//

#include "../ABIInfo.h"
#include "../CIRGenModule.h"
#include "../TargetInfo.h"

#include "clang/CIR/Dialect/IR/CIRTypes.h"

using namespace clang;
using namespace clang::CIRGen;

namespace {

class NVPTXABIInfo : public ABIInfo {
public:
  NVPTXABIInfo(CIRGenTypes &cgt) : ABIInfo(cgt) {}
};

class NVPTXTargetCIRGenInfo : public TargetCIRGenInfo {
public:
  NVPTXTargetCIRGenInfo(CIRGenTypes &cgt)
      : TargetCIRGenInfo(std::make_unique<NVPTXABIInfo>(cgt)) {}

  void setTargetAttributes(const clang::Decl *decl, mlir::Operation *global,
                           CIRGenModule &cgm) const override {
    auto globalValue = mlir::dyn_cast<cir::CIRGlobalValueInterface>(global);
    if (globalValue && globalValue.isDeclaration())
      return;

    const auto *vd = dyn_cast_or_null<VarDecl>(decl);
    if (vd) {
      if (cgm.getLangOpts().CUDA) {
        if (vd->getType()->isCUDADeviceBuiltinSurfaceType() ||
            vd->getType()->isCUDADeviceBuiltinTextureType())
          assert(!cir::MissingFeatures::emitNVVMMetadata());
        return;
      }
    }

    const auto *fd = dyn_cast_or_null<FunctionDecl>(decl);
    if (!fd)
      return;

    auto func = mlir::cast<cir::FuncOp>(global);

    // Perform special handling in OpenCL/CUDA mode.
    if (cgm.getLangOpts().OpenCL || cgm.getLangOpts().CUDA) {
      // Use function attributes to check for kernel functions. By default, all
      // functions are device functions.
      if (fd->hasAttr<DeviceKernelAttr>() || fd->hasAttr<CUDAGlobalAttr>()) {
        // OpenCL/CUDA kernel functions get kernel metadata. Kernel functions
        // are also not subject to inlining.
        func.setInlineKind(cir::InlineKind::NoInline);
        if (fd->hasAttr<CUDAGlobalAttr>()) {
          func.setCallingConv(cir::CallingConv::PTXKernel);
          assert(!cir::MissingFeatures::opFuncParameterAttributes());
        }
        if (fd->hasAttr<CUDALaunchBoundsAttr>())
          assert(!cir::MissingFeatures::handleCUDALaunchBoundsAttr());
      }
    }
  }

  mlir::Type getCUDADeviceBuiltinSurfaceDeviceType() const override {
    // On the device side, surface reference is represented as an object handle
    // in 64-bit integer.
    return cir::IntType::get(&getABIInfo().cgt.getMLIRContext(), 64,
                             /*isSigned=*/true);
  }
};

} // namespace

std::unique_ptr<TargetCIRGenInfo>
clang::CIRGen::createNVPTXTargetCIRGenInfo(CIRGenTypes &cgt) {
  return std::make_unique<NVPTXTargetCIRGenInfo>(cgt);
}
