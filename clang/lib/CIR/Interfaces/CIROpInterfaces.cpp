//====- CIROpInterfaces.cpp - Interface to AST Attributes ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the interface to CIR operations.
//
//===----------------------------------------------------------------------===//
#include "clang/CIR/Interfaces/CIROpInterfaces.h"

using namespace cir;

/// Include the generated type qualifiers interfaces.
#include "clang/CIR/Interfaces/CIROpInterfaces.cpp.inc"

#include "clang/CIR/MissingFeatures.h"

bool CIRGlobalValueInterface::canBenefitFromLocalAlias() {
  assert(!cir::MissingFeatures::supportIFuncAttr());
  assert(!cir::MissingFeatures::supportVisibility());
  assert(!cir::MissingFeatures::supportComdat());
  return false;
}
