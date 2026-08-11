//===---- SPIRV.cpp - SPIR-V-specific CIR CodeGen -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides SPIR-V-specific CIR CodeGen logic for function attributes.
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

class SPIRVABIInfo : public ABIInfo {
public:
  SPIRVABIInfo(CIRGenTypes &cgt) : ABIInfo(cgt) {}

  cir::VectorType getOptimalVectorMemoryType(
      cir::VectorType type, const clang::LangOptions &langOpts) const override {
    // Logical SPIR-V does not describe the underlying hardware or guarantee
    // that a smaller vector occupies storage for a larger vector.
    if (cgt.getCGModule().getTarget().getTriple().isSPIRVLogical())
      return type;
    return ABIInfo::getOptimalVectorMemoryType(type, langOpts);
  }
};

class SPIRVTargetCIRGenInfo : public TargetCIRGenInfo {
public:
  // TODO(CIR): Select an AMDGPU-specific ABIInfo for AMD-vendor SPIR-V once
  // CIR supports that target.
  SPIRVTargetCIRGenInfo(CIRGenTypes &cgt)
      : TargetCIRGenInfo(std::make_unique<SPIRVABIInfo>(cgt)) {}

  void setTargetAttributes(const clang::Decl *decl, mlir::Operation *global,
                           CIRGenModule &cgm) const override {
    auto globalValue = mlir::cast<cir::CIRGlobalValueInterface>(global);
    if (globalValue.isDeclaration())
      return;

    const auto *fd = dyn_cast_or_null<FunctionDecl>(decl);
    if (!fd)
      return;

    if (cgm.getLangOpts().OpenCL &&
        DeviceKernelAttr::isOpenCLSpelling(fd->getAttr<DeviceKernelAttr>())) {
      auto func = mlir::cast<cir::FuncOp>(global);
      func.setCallingConv(cir::CallingConv::SpirKernel);
    }
  }
};

} // namespace

std::unique_ptr<TargetCIRGenInfo>
clang::CIRGen::createSPIRVTargetCIRGenInfo(CIRGenTypes &cgt) {
  return std::make_unique<SPIRVTargetCIRGenInfo>(cgt);
}
