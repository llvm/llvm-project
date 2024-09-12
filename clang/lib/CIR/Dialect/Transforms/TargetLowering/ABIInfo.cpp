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
#include "clang/CIR/Dialect/IR/CIRDataLayout.h"

namespace mlir {
namespace cir {

// Pin the vtable to this file.
ABIInfo::~ABIInfo() = default;

CIRCXXABI &ABIInfo::getCXXABI() const { return LT.getCXXABI(); }

CIRLowerContext &ABIInfo::getContext() const { return LT.getContext(); }

const clang::TargetInfo &ABIInfo::getTarget() const { return LT.getTarget(); }

const ::cir::CIRDataLayout &ABIInfo::getDataLayout() const {
  return LT.getDataLayout();
}

bool ABIInfo::isPromotableIntegerTypeForABI(Type Ty) const {
  if (getContext().isPromotableIntegerType(Ty))
    return true;

  assert(!::cir::MissingFeatures::fixedWidthIntegers());

  return false;
}

} // namespace cir
} // namespace mlir
