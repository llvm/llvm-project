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

using namespace cir;

/// Include the generated type qualifiers interfaces.
#include "clang/CIR/Interfaces/CIROpInterfaces.cpp.inc"

#include "clang/CIR/MissingFeatures.h"

bool CIRGlobalValueInterface::hasDefaultVisibility() {
  assert(!cir::MissingFeatures::hiddenVisibility());
  assert(!cir::MissingFeatures::protectedVisibility());
  return isPublic() || isPrivate();
}

bool CIRGlobalValueInterface::canBenefitFromLocalAlias() {
  assert(!cir::MissingFeatures::supportIFuncAttr());
  // hasComdat here should be isDeduplicateComdat, but as far as clang codegen
  // is concerned, there is no case for Comdat::NoDeduplicate as all comdat
  // would be Comdat::Any or Comdat::Largest (in the case of MS ABI). And CIRGen
  // wouldn't even generate Comdat::Largest comdat as it tries to leave ABI
  // specifics to LLVM lowering stage, thus here we don't need test Comdat
  // selectionKind.
  return hasDefaultVisibility() && hasExternalLinkage() && !isDeclaration() &&
         !hasComdat();
  return false;
}
