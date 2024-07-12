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
#include "llvm/Support/ErrorHandling.h"

namespace mlir {
namespace cir {

namespace {

class ItaniumCXXABI : public CIRCXXABI {

protected:
  bool UseARMMethodPtrABI;
  bool UseARMGuardVarABI;
  bool Use32BitVTableOffsetABI;

public:
  ItaniumCXXABI(LowerModule &LM, bool UseARMMethodPtrABI = false,
                bool UseARMGuardVarABI = false)
      : CIRCXXABI(LM), UseARMMethodPtrABI(UseARMMethodPtrABI),
        UseARMGuardVarABI(UseARMGuardVarABI), Use32BitVTableOffsetABI(false) {}

  bool classifyReturnType(LowerFunctionInfo &FI) const override;
};

} // namespace

bool ItaniumCXXABI::classifyReturnType(LowerFunctionInfo &FI) const {
  const StructType RD = dyn_cast<StructType>(FI.getReturnType());
  if (!RD)
    return false;

  // If C++ prohibits us from making a copy, return by address.
  if (::cir::MissingFeatures::recordDeclCanPassInRegisters())
    llvm_unreachable("NYI");

  return false;
}

CIRCXXABI *CreateItaniumCXXABI(LowerModule &LM) {
  switch (LM.getCXXABIKind()) {
  // Note that AArch64 uses the generic ItaniumCXXABI class since it doesn't
  // include the other 32-bit ARM oddities: constructor/destructor return values
  // and array cookies.
  case clang::TargetCXXABI::GenericAArch64:
  case clang::TargetCXXABI::AppleARM64:
    // TODO: this isn't quite right, clang uses AppleARM64CXXABI which inherits
    // from ARMCXXABI. We'll have to follow suit.
    assert(!::cir::MissingFeatures::appleArm64CXXABI());
    return new ItaniumCXXABI(LM, /*UseARMMethodPtrABI=*/true,
                             /*UseARMGuardVarABI=*/true);

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
