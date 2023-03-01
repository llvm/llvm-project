//===- HLFIRDialect.h - High Level Fortran IR dialect -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the HLFIR dialect that models Fortran expressions and
// assignments without requiring storage allocation and manipulations.
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_HLFIR_HLFIRDIALECT_H
#define FORTRAN_OPTIMIZER_HLFIR_HLFIRDIALECT_H

#include "flang/Optimizer/Dialect/FIRType.h"
#include "mlir/IR/Dialect.h"

namespace hlfir {
/// Is this a type that can be used for an HLFIR variable ?
bool isFortranVariableType(mlir::Type);
bool isFortranScalarCharacterType(mlir::Type);
bool isFortranScalarCharacterExprType(mlir::Type);
} // namespace hlfir

#include "flang/Optimizer/HLFIR/HLFIRDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "flang/Optimizer/HLFIR/HLFIRTypes.h.inc"

#define GET_ATTRDEF_CLASSES
#include "flang/Optimizer/HLFIR/HLFIRAttributes.h.inc"

namespace hlfir {
/// Get the element type of a Fortran entity type.
inline mlir::Type getFortranElementType(mlir::Type type) {
  type = fir::unwrapSequenceType(
      fir::unwrapPassByRefType(fir::unwrapRefType(type)));
  if (auto exprType = type.dyn_cast<hlfir::ExprType>())
    return exprType.getEleTy();
  if (auto boxCharType = type.dyn_cast<fir::BoxCharType>())
    return boxCharType.getEleTy();
  return type;
}

/// If this is the type of a Fortran array entity, get the related
/// fir.array type. Otherwise, returns the Fortran element typeof the entity.
inline mlir::Type getFortranElementOrSequenceType(mlir::Type type) {
  type = fir::unwrapPassByRefType(fir::unwrapRefType(type));
  if (auto exprType = type.dyn_cast<hlfir::ExprType>()) {
    if (exprType.isArray())
      return fir::SequenceType::get(exprType.getShape(), exprType.getEleTy());
    return exprType.getEleTy();
  }
  if (auto boxCharType = type.dyn_cast<fir::BoxCharType>())
    return boxCharType.getEleTy();
  return type;
}

/// Is this a fir.box or fir.class address type?
inline bool isBoxAddressType(mlir::Type type) {
  type = fir::dyn_cast_ptrEleTy(type);
  return type && type.isa<fir::BaseBoxType>();
}

/// Is this a fir.box or fir.class address or value type?
inline bool isBoxAddressOrValueType(mlir::Type type) {
  return fir::unwrapRefType(type).isa<fir::BaseBoxType>();
}

bool isFortranScalarNumericalType(mlir::Type);
bool isFortranNumericalArrayObject(mlir::Type);
bool isFortranNumericalOrLogicalArrayObject(mlir::Type);
bool isFortranArrayObject(mlir::Type);
bool isPassByRefOrIntegerType(mlir::Type);
bool isI1Type(mlir::Type);
// scalar i1 or logical, or sequence of logical (via (boxed?) array or expr)
bool isMaskArgument(mlir::Type);

/// If an expression's extents are known at compile time, generate a fir.shape
/// for this expression. Otherwise return {}
mlir::Value genExprShape(mlir::OpBuilder &builder, const mlir::Location &loc,
                         const hlfir::ExprType &expr);

} // namespace hlfir

#endif // FORTRAN_OPTIMIZER_HLFIR_HLFIRDIALECT_H
