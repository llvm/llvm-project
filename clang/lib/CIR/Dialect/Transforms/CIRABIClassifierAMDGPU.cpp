//===- CIRABIClassifierAMDGPU.cpp - AMDGPU classification -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the AMDGPU classifier, which classifies with the LLVM
// ABI Lowering Library's AMDGPU target.
//
//===----------------------------------------------------------------------===//

#include "CIRABIClassifier.h"

#include "clang/CIR/Dialect/IR/CIRDialect.h"

using namespace mlir;
using namespace mlir::abi;
using namespace cir;

namespace {

class AMDGPUABIClassifier final : public CIRABIClassifierCommon {
public:
  AMDGPUABIClassifier(const mlir::DataLayout &dl, mlir::ModuleOp modOp)
      : CIRABIClassifierCommon(dl, modOp) {}

  std::optional<ArgClassification> classifyVAArg(cir::VAArgOp vaArg) override {
    vaArg->emitOpError()
        << "not yet implemented in CallConvLowering for AMDGPU";
    return std::nullopt;
  }

protected:
  /// AMDGPU has no per-function feature that changes its ABI rules.
  const llvm::abi::TargetInfo &getTargetInfo(cir::FuncOp) override;

private:
  std::unique_ptr<llvm::abi::TargetInfo> target;
};

const llvm::abi::TargetInfo &AMDGPUABIClassifier::getTargetInfo(cir::FuncOp) {
  if (!target) {
    // HIP passes a generic pointer kernel argument as a global one.
    auto langOpts = modOp->getAttrOfType<cir::LoweringLangOptionsAttr>(
        cir::CIRDialect::getLoweringLangOptionsAttrName());
    target = llvm::abi::createAMDGPUTargetInfo(
        typeMapper.getTypeBuilder(),
        /*CoerceGenericPtrArgToGlobal=*/langOpts && langOpts.getHip());
  }
  return *target;
}

} // namespace

std::unique_ptr<CIRABIClassifier>
cir::createAMDGPUABIClassifier(const mlir::DataLayout &dl,
                               mlir::ModuleOp modOp) {
  return std::make_unique<AMDGPUABIClassifier>(dl, modOp);
}
