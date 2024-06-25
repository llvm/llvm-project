//====- CIROpInterfaces.cpp - Interface to AST Attributes ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "clang/CIR/Interfaces/CIROpInterfaces.h"

#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir::cir;

/// Include the generated type qualifiers interfaces.
#include "clang/CIR/Interfaces/CIROpInterfaces.cpp.inc"

#include "clang/CIR/MissingFeatures.h"

bool CIRGlobalValueInterface::canBenefitFromLocalAlias() const {
  assert(!::cir::MissingFeatures::supportIFuncAttr());
  assert(!::cir::MissingFeatures::setComdat());
  return false;
}
