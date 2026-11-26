//===---- AArch64.cpp - AArch64-specific CIR CodeGen ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides AArch64-specific CIR CodeGen logic.
//
//===----------------------------------------------------------------------===//

#include "ABIInfo.h"
#include "CIRGenFunction.h"
#include "CIRGenModule.h"
#include "TargetInfo.h"
#include "clang/AST/Decl.h"
#include "clang/CIR/MissingFeatures.h"
#include "clang/CodeGenUtils/TargetUtils.h"

using namespace clang;
using namespace clang::CIRGen;

namespace {

class AArch64ABIInfo : public ABIInfo {
public:
  AArch64ABIInfo(CIRGenTypes &cgt) : ABIInfo(cgt) {}
};

class AArch64TargetCIRGenInfo : public TargetCIRGenInfo {
public:
  AArch64TargetCIRGenInfo(CIRGenTypes &cgt)
      : TargetCIRGenInfo(std::make_unique<AArch64ABIInfo>(cgt)) {}

  void setTargetAttributes(const Decl *d, mlir::Operation *gv,
                           CIRGenModule &cgm) const override {
    auto fn = mlir::dyn_cast<cir::FuncOp>(gv);
    if (!fn)
      return;
    assert(!cir::MissingFeatures::branchProtection());
    assert(!cir::MissingFeatures::pointerAuthentication());
  }

  bool isScalarizableAsmOperand(CIRGenFunction &cgf,
                                mlir::Type ty) const override {
    if (cgf.getTarget().hasFeature("ls64")) {
      cgf.cgm.errorNYI("AArch64 LS64 scalarizable asm operand");
      return true;
    }
    return TargetCIRGenInfo::isScalarizableAsmOperand(cgf, ty);
  }

  bool wouldInliningViolateFunctionCallABI(
      const FunctionDecl *caller, const FunctionDecl *callee) const override;
};

} // namespace

bool AArch64TargetCIRGenInfo::wouldInliningViolateFunctionCallABI(
    const FunctionDecl *caller, const FunctionDecl *callee) const {
  return caller && callee &&
         CodeGenUtils::getArmSMEInlinability(caller, callee) !=
             CodeGenUtils::ArmSMEInlinability::Ok;
}

std::unique_ptr<TargetCIRGenInfo>
clang::CIRGen::createAArch64TargetCIRGenInfo(CIRGenTypes &cgt) {
  return std::make_unique<AArch64TargetCIRGenInfo>(cgt);
}
