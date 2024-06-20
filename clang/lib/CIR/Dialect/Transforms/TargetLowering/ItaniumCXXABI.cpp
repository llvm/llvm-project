//===------- ItaniumCXXABI.cpp - Emit CIR code Itanium-specific code  -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides CIR lowering logic targeting the Itanium C++ ABI. The class in
// this file generates structures that follow the Itanium C++ ABI, which is
// documented at:
//  https://itanium-cxx-abi.github.io/cxx-abi/abi.html
//  https://itanium-cxx-abi.github.io/cxx-abi/abi-eh.html
//
// It also supports the closely-related ARM ABI, documented at:
// https://developer.arm.com/documentation/ihi0041/g/
//
// This file partially mimics clang/lib/CodeGen/ItaniumCXXABI.cpp. The queries
// are adapted to operate on the CIR dialect, however.
//
//===----------------------------------------------------------------------===//

#include "CIRCXXABI.h"
#include "LowerModule.h"

namespace mlir {
namespace cir {

namespace {

class ItaniumCXXABI : public CIRCXXABI {

public:
  ItaniumCXXABI(LowerModule &LM) : CIRCXXABI(LM) {}

  bool classifyReturnType(LowerFunctionInfo &FI) const override;
};

} // namespace

bool ItaniumCXXABI::classifyReturnType(LowerFunctionInfo &FI) const {
  const StructType RD = FI.getReturnType().dyn_cast<StructType>();
  if (!RD)
    return false;

  // If C++ prohibits us from making a copy, return by address.
  if (::cir::MissingFeatures::recordDeclCanPassInRegisters())
    llvm_unreachable("NYI");

  return false;
}

CIRCXXABI *CreateItaniumCXXABI(LowerModule &LM) {
  switch (LM.getCXXABIKind()) {
  case clang::TargetCXXABI::GenericItanium:
    return new ItaniumCXXABI(LM);

  case clang::TargetCXXABI::Microsoft:
    llvm_unreachable("Microsoft ABI is not Itanium-based");
  default:
    llvm_unreachable("NYI");
  }

  llvm_unreachable("bad ABI kind");
}

} // namespace cir
} // namespace mlir

// FIXME(cir): Merge this into the CIRCXXABI class above.
class LoweringPrepareItaniumCXXABI : public cir::LoweringPrepareCXXABI {
public:
  mlir::Value lowerDynamicCast(cir::CIRBaseBuilderTy &builder,
                               clang::ASTContext &astCtx,
                               mlir::cir::DynamicCastOp op) override;
  mlir::Value lowerVAArg(cir::CIRBaseBuilderTy &builder, mlir::cir::VAArgOp op,
                         const cir::CIRDataLayout &datalayout) override;
};
