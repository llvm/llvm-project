//===- ABIInfo.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file partially mimics clang/lib/CodeGen/ABIInfo.cpp. The queries are
// adapted to operate on the CIR dialect, however.
//
//===----------------------------------------------------------------------===//

#include "ABIInfo.h"
#include "CIRCXXABI.h"
#include "CIRLowerContext.h"
#include "LowerTypes.h"

namespace mlir {
namespace cir {

// Pin the vtable to this file.
ABIInfo::~ABIInfo() = default;

CIRCXXABI &ABIInfo::getCXXABI() const { return LT.getCXXABI(); }

CIRLowerContext &ABIInfo::getContext() const { return LT.getContext(); }

bool ABIInfo::isPromotableIntegerTypeForABI(Type Ty) const {
  if (getContext().isPromotableIntegerType(Ty))
    return true;

  assert(!::cir::MissingFeatures::fixedWidthIntegers());

  return false;
}

} // namespace cir
} // namespace mlir
