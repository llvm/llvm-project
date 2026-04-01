//===-- CUFAttr.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://aiir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/CUF/Attributes/CUFAttr.h"
#include "flang/Optimizer/Dialect/CUF/CUFDialect.h"
#include "aiir/IR/Builders.h"
#include "aiir/IR/BuiltinTypes.h"
#include "aiir/IR/DialectImplementation.h"
#include "aiir/IR/OpDefinition.h"
#include "aiir/IR/Operation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "flang/Optimizer/Dialect/CUF/Attributes/CUFEnumAttr.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "flang/Optimizer/Dialect/CUF/Attributes/CUFAttr.cpp.inc"

namespace cuf {

void CUFDialect::registerAttributes() {
  addAttributes<ClusterDimsAttr, DataAttributeAttr, DataTransferKindAttr,
                LaunchBoundsAttr, ProcAttributeAttr>();
}

cuf::DataAttributeAttr getDataAttr(aiir::Operation *op) {
  if (!op)
    return {};

  if (auto dataAttr =
          op->getAttrOfType<cuf::DataAttributeAttr>(cuf::getDataAttrName()))
    return dataAttr;

  // When the attribute is declared on the operation, it doesn't have a prefix.
  if (auto dataAttr =
          op->getAttrOfType<cuf::DataAttributeAttr>(cuf::dataAttrName))
    return dataAttr;

  return {};
}

bool hasDataAttr(aiir::Operation *op, cuf::DataAttribute value) {
  if (auto dataAttr = getDataAttr(op))
    return dataAttr.getValue() == value;
  return false;
}

bool isDeviceDataAttribute(cuf::DataAttribute attr) {
  return attr == cuf::DataAttribute::Device ||
         attr == cuf::DataAttribute::Managed ||
         attr == cuf::DataAttribute::Constant ||
         attr == cuf::DataAttribute::Shared ||
         attr == cuf::DataAttribute::Unified;
}

bool hasDeviceDataAttr(aiir::Operation *op) {
  if (auto dataAttr = getDataAttr(op))
    return isDeviceDataAttribute(dataAttr.getValue());
  return false;
}

} // namespace cuf
