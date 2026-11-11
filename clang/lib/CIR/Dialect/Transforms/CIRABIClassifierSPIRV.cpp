//===- CIRABIClassifierSPIRV.cpp - SPIR-V classification ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the SPIR-V classifier, which classifies with the LLVM
// ABI Lowering Library's SPIR-V target.
//
//===----------------------------------------------------------------------===//

#include "CIRABIClassifier.h"
#include "TargetLowering/LowerModule.h"

#include "clang/CIR/Dialect/IR/CIRDialect.h"

using namespace mlir;
using namespace mlir::abi;
using namespace cir;

namespace {

class SPIRVABIClassifier final : public CIRABIClassifierCommon {
public:
  SPIRVABIClassifier(const mlir::DataLayout &dl, mlir::ModuleOp modOp)
      : CIRABIClassifierCommon(dl, modOp) {}

  mlir::FailureOr<ArgClassification>
  classifyVAArg(cir::VAArgOp vaArg) override {
    vaArg->emitOpError()
        << "not yet implemented in CallConvLowering for SPIR-V";
    return mlir::failure();
  }

protected:
  /// SPIR-V has no per-function feature that changes its ABI rules.
  const llvm::abi::TargetInfo &getTargetInfo(cir::FuncOp) override;

private:
  std::unique_ptr<llvm::abi::TargetInfo> target;
};

const llvm::abi::TargetInfo &SPIRVABIClassifier::getTargetInfo(cir::FuncOp) {
  if (!target) {
    // Classic codegen passes the target address space of LangAS::Default and
    // LangOptions::isTargetDevice().
    // LowerModule rebuilds both from the module's attributes, so a reloaded
    // .cir classifies the same way it was compiled. Without cir.triple
    // there is no target, so use 0 as default address space.
    std::unique_ptr<cir::LowerModule> lowerModule =
        cir::createLowerModule(modOp);
    assert(lowerModule && "LowerModule shouldn't be null");
    unsigned defaultAddrSpace =
        lowerModule->hasTarget()
            ? lowerModule->getTarget().getTargetAddressSpace(
                  clang::LangAS::Default)
            : 0;

    target = llvm::abi::createSPIRVTargetInfo(
        typeMapper.getTypeBuilder(), defaultAddrSpace,
        /*KernelPassAggregatesIndirect=*/
        lowerModule->getLangOpts().isTargetDevice());
  }
  return *target;
}

} // namespace

std::unique_ptr<CIRABIClassifier>
cir::createSPIRVABIClassifier(const mlir::DataLayout &dl,
                              mlir::ModuleOp modOp) {
  return std::make_unique<SPIRVABIClassifier>(dl, modOp);
}
