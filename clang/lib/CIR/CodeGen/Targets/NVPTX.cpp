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
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/NVVMAttributes.h"

using namespace clang;
using namespace clang::CIRGen;

namespace {

/// Handle the launch_bounds attribute, which maps onto the nvvm.maxntid,
/// nvvm.minctasm and nvvm.maxclusterrank function attributes.
static void handleCUDALaunchBoundsAttr(const CUDALaunchBoundsAttr *attr,
                                       cir::FuncOp func, CIRGenModule &cgm,
                                       CIRGenBuilderTy &builder) {
  auto setNVVMAttr = [&](llvm::StringRef name, const llvm::APSInt &value) {
    func->setAttr(("cir." + name).str(),
                  builder.getStringAttr(llvm::utostr(value.getExtValue())));
  };

  llvm::APSInt maxThreads(32);
  maxThreads =
      attr->getMaxThreads()->EvaluateKnownConstInt(cgm.getASTContext());
  if (maxThreads > 0)
    setNVVMAttr(llvm::NVVMAttr::MaxNTID, maxThreads);

  // min and max blocks is an optional argument for CUDALaunchBoundsAttr. If it
  // was not specified in __launch_bounds__ or if the user specified a 0 value,
  // we don't have to add a PTX directive.
  if (attr->getMinBlocks()) {
    llvm::APSInt minBlocks(32);
    minBlocks =
        attr->getMinBlocks()->EvaluateKnownConstInt(cgm.getASTContext());
    if (minBlocks > 0)
      setNVVMAttr(llvm::NVVMAttr::MinCTASm, minBlocks);
  }

  if (attr->getMaxBlocks()) {
    llvm::APSInt maxBlocks(32);
    maxBlocks =
        attr->getMaxBlocks()->EvaluateKnownConstInt(cgm.getASTContext());
    if (maxBlocks > 0)
      setNVVMAttr(llvm::NVVMAttr::MaxClusterRank, maxBlocks);
  }
}

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
        if (const auto *attr = fd->getAttr<CUDALaunchBoundsAttr>())
          handleCUDALaunchBoundsAttr(attr, func, cgm, cgm.getBuilder());
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
