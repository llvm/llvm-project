//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the types in the CIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_DIALECT_IR_CIRTYPES_H
#define CLANG_CIR_DIALECT_IR_CIRTYPES_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "clang/Basic/AddressSpaces.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Interfaces/CIRTypeInterfaces.h"

namespace cir {

namespace detail {
struct RecordTypeStorage;
} // namespace detail

bool isValidFundamentalIntWidth(unsigned width);

/// Returns true if the type is a CIR sized type.
///
/// Types are sized if they implement SizedTypeInterface and
/// return true from its method isSized.
///
/// Unsized types are those that do not have a size, such as
/// void, or abstract types.
bool isSized(mlir::Type ty);

} // namespace cir

//===----------------------------------------------------------------------===//
// CIR Dialect Tablegen'd Types
//===----------------------------------------------------------------------===//

namespace cir {

#include "clang/CIR/Dialect/IR/CIRTypeConstraints.h.inc"

} // namespace cir

#define GET_TYPEDEF_CLASSES
#include "clang/CIR/Dialect/IR/CIROpsTypes.h.inc"

#endif // CLANG_CIR_DIALECT_IR_CIRTYPES_H
