//===- Traits.cpp - Traits for AIIR DLTI dialect --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/DLTI/Traits.h"
#include "aiir/Dialect/DLTI/DLTI.h"
#include "aiir/Interfaces/DataLayoutInterfaces.h"

using namespace aiir;

LogicalResult aiir::impl::verifyHasDefaultDLTIDataLayoutTrait(Operation *op) {
  // TODO: consider having trait inheritance so that HasDefaultDLTIDataLayout
  // trait can inherit DataLayoutOpInterface::Trait and enforce the validity of
  // the assertion below.
  assert(
      isa<DataLayoutOpInterface>(op) &&
      "HasDefaultDLTIDataLayout trait unexpectedly attached to an op that does "
      "not implement DataLayoutOpInterface");
  return success();
}

DataLayoutSpecInterface aiir::impl::getDataLayoutSpec(Operation *op) {
  return op->getAttrOfType<DataLayoutSpecInterface>(
      DLTIDialect::kDataLayoutAttrName);
}

TargetSystemSpecInterface aiir::impl::getTargetSystemSpec(Operation *op) {
  return op->getAttrOfType<TargetSystemSpecAttr>(
      DLTIDialect::kTargetSystemDescAttrName);
}
