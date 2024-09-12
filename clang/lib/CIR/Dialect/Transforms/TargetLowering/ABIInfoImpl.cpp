//===--- ABIInfoImpl.cpp - Encapsulate calling convention details ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file partially mimics clang/lib/CodeGen/ABIInfoImpl.cpp. The queries are
// adapted to operate on the CIR dialect, however.
//
//===----------------------------------------------------------------------===//

#include "ABIInfo.h"
#include "CIRCXXABI.h"
#include "LowerFunction.h"
#include "LowerFunctionInfo.h"
#include "clang/CIR/MissingFeatures.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir {
namespace cir {

bool classifyReturnType(const CIRCXXABI &CXXABI, LowerFunctionInfo &FI,
                        const ABIInfo &Info) {
  Type Ty = FI.getReturnType();

  if (const auto RT = dyn_cast<StructType>(Ty)) {
    assert(!::cir::MissingFeatures::isCXXRecordDecl());
  }

  return CXXABI.classifyReturnType(FI);
}

bool isAggregateTypeForABI(Type T) {
  assert(!::cir::MissingFeatures::functionMemberPointerType());
  return !LowerFunction::hasScalarEvaluationKind(T);
}

Type useFirstFieldIfTransparentUnion(Type Ty) {
  if (auto RT = dyn_cast<StructType>(Ty)) {
    if (RT.isUnion())
      llvm_unreachable("NYI");
  }
  return Ty;
}

CIRCXXABI::RecordArgABI getRecordArgABI(const StructType RT,
                                        CIRCXXABI &CXXABI) {
  if (::cir::MissingFeatures::typeIsCXXRecordDecl()) {
    llvm_unreachable("NYI");
  }
  return CXXABI.getRecordArgABI(RT);
}

} // namespace cir
} // namespace mlir
