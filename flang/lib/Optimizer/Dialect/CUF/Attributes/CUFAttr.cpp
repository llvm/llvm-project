//===-- CUFAttr.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/CUF/Attributes/CUFAttr.h"
#include "flang/Optimizer/Dialect/CUF/CUFDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/TypeSwitch.h"

#include "flang/Optimizer/Dialect/CUF/Attributes/CUFEnumAttr.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "flang/Optimizer/Dialect/CUF/Attributes/CUFAttr.cpp.inc"

namespace cuf {

void CUFDialect::registerAttributes() {
  addAttributes<ClusterDimsAttr, DataAttributeAttr, DataTransferKindAttr,
                LaunchBoundsAttr, ProcAttributeAttr>();
}

} // namespace cuf
