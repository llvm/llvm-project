//===--- LowerTypes.cpp - Type translation to target-specific types -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file partially mimics clang/lib/CodeGen/CodeGenTypes.cpp. The queries
// are adapted to operate on the CIR dialect, however.
//
//===----------------------------------------------------------------------===//

#include "LowerTypes.h"
#include "CIRToCIRArgMapping.h"
#include "LowerModule.h"
#include "mlir/Support/LLVM.h"
#include "clang/CIR/ABIArgInfo.h"
#include "clang/CIR/MissingFeatures.h"
#include "llvm/Support/ErrorHandling.h"

using namespace ::mlir::cir;

unsigned LowerTypes::clangCallConvToLLVMCallConv(clang::CallingConv CC) {
  switch (CC) {
  case clang::CC_C:
    return llvm::CallingConv::C;
  default:
    llvm_unreachable("calling convention NYI");
  }
}

LowerTypes::LowerTypes(LowerModule &LM, StringRef DLString)
    : LM(LM), context(LM.getContext()), Target(LM.getTarget()),
      CXXABI(LM.getCXXABI()),
      TheABIInfo(LM.getTargetLoweringInfo().getABIInfo()),
      mlirContext(LM.getMLIRContext()), DL(DLString, LM.getModule()) {}

/// Return the ABI-specific function type for a CIR function type.
FuncType LowerTypes::getFunctionType(const LowerFunctionInfo &FI) {

  mlir::Type resultType = {};
  const ::cir::ABIArgInfo &retAI = FI.getReturnInfo();
  switch (retAI.getKind()) {
  case ::cir::ABIArgInfo::Ignore:
    resultType = VoidType::get(getMLIRContext());
    break;
  default:
    llvm_unreachable("Missing ABIArgInfo::Kind");
  }

  CIRToCIRArgMapping IRFunctionArgs(getContext(), FI, true);
  SmallVector<Type, 8> ArgTypes(IRFunctionArgs.totalIRArgs());

  // Add type for sret argument.
  assert(!::cir::MissingFeatures::sretArgs());

  // Add type for inalloca argument.
  assert(!::cir::MissingFeatures::inallocaArgs());

  // Add in all of the required arguments.
  unsigned ArgNo = 0;
  LowerFunctionInfo::const_arg_iterator it = FI.arg_begin(),
                                        ie = it + FI.getNumRequiredArgs();
  for (; it != ie; ++it, ++ArgNo) {
    llvm_unreachable("NYI");
  }

  return FuncType::get(getMLIRContext(), ArgTypes, resultType, FI.isVariadic());
}
