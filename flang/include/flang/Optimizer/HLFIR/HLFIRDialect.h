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
bool isFortranArrayCharacterExprType(mlir::Type);
} // namespace hlfir

#include "flang/Optimizer/HLFIR/HLFIRDialect.h.inc"

#include "flang/Optimizer/HLFIR/HLFIREnums.h.inc"

#define GET_TYPEDEF_CLASSES
#include "flang/Optimizer/HLFIR/HLFIRTypes.h.inc"

#define GET_ATTRDEF_CLASSES
#include "flang/Optimizer/HLFIR/HLFIRAttributes.h.inc"

namespace hlfir {
/// Get the element type of a Fortran entity type.
inline mlir::Type getFortranElementType(mlir::Type type) {
  type = fir::unwrapSequenceType(
      fir::unwrapPassByRefType(fir::unwrapRefType(type)));
  if (auto exprType = mlir::dyn_cast<hlfir::ExprType>(type))
    return exprType.getEleTy();
  if (auto boxCharType = mlir::dyn_cast<fir::BoxCharType>(type))
    return boxCharType.getEleTy();
  return type;
}

/// If this is the type of a Fortran array entity, get the related
/// fir.array type. Otherwise, returns the Fortran element typeof the entity.
inline mlir::Type getFortranElementOrSequenceType(mlir::Type type) {
  type = fir::unwrapPassByRefType(fir::unwrapRefType(type));
  if (auto exprType = mlir::dyn_cast<hlfir::ExprType>(type)) {
    if (exprType.isArray())
      return fir::SequenceType::get(exprType.getShape(), exprType.getEleTy());
    return exprType.getEleTy();
  }
  if (auto boxCharType = mlir::dyn_cast<fir::BoxCharType>(type))
    return boxCharType.getEleTy();
  return type;
}

/// Is this a fir.box or fir.class address type?
inline bool isBoxAddressType(mlir::Type type) {
  type = fir::dyn_cast_ptrEleTy(type);
  return type && mlir::isa<fir::BaseBoxType>(type);
}

/// Is this a fir.box or fir.class address or value type?
inline bool isBoxAddressOrValueType(mlir::Type type) {
  return mlir::isa<fir::BaseBoxType>(fir::unwrapRefType(type));
}

inline bool isPolymorphicType(mlir::Type type) {
  if (auto exprType = mlir::dyn_cast<hlfir::ExprType>(type))
    return exprType.isPolymorphic();
  return fir::isPolymorphicType(type);
}

/// Is this an SSA value type for the value of a Fortran procedure
/// designator ?
inline bool isFortranProcedureValue(mlir::Type type) {
  return mlir::isa<fir::BoxProcType>(type) ||
         (mlir::isa<mlir::TupleType>(type) &&
          fir::isCharacterProcedureTuple(type, /*acceptRawFunc=*/false));
}

/// Is this an SSA value type for the value of a Fortran expression?
inline bool isFortranValueType(mlir::Type type) {
  return mlir::isa<hlfir::ExprType>(type) || fir::isa_trivial(type) ||
         isFortranProcedureValue(type);
}

/// Is this the value of a Fortran expression in an SSA value form?
inline bool isFortranValue(mlir::Value value) {
  return isFortranValueType(value.getType());
}

/// Is this a Fortran variable?
/// Note that by "variable", it must be understood that the mlir::Value is
/// a memory value of a storage that can be reason about as a Fortran object
/// (its bounds, shape, and type parameters, if any, are retrievable).
/// This does not imply that the mlir::Value points to a variable from the
/// original source or can be legally defined: temporaries created to store
/// expression values are considered to be variables, and so are PARAMETERs
/// global constant address.
inline bool isFortranEntity(mlir::Value value) {
  return isFortranValue(value) || isFortranVariableType(value.getType());
}

bool isFortranScalarNumericalType(mlir::Type);
bool isFortranNumericalArrayObject(mlir::Type);
bool isFortranNumericalOrLogicalArrayObject(mlir::Type);
bool isFortranArrayObject(mlir::Type);
bool isFortranLogicalArrayObject(mlir::Type);
bool isPassByRefOrIntegerType(mlir::Type);
bool isI1Type(mlir::Type);
// scalar i1 or logical, or sequence of logical (via (boxed?) array or expr)
bool isMaskArgument(mlir::Type);
bool isPolymorphicObject(mlir::Type);

/// If an expression's extents are known at compile time, generate a fir.shape
/// for this expression. Otherwise return {}
mlir::Value genExprShape(mlir::OpBuilder &builder, const mlir::Location &loc,
                         const hlfir::ExprType &expr);

/// Return true iff `ty` may have allocatable component.
/// TODO: this actually belongs to FIRType.cpp, but the method's implementation
/// depends on HLFIRDialect component. FIRType.cpp itself is part of FIRDialect
/// that cannot depend on HLFIRBuilder (there will be a cyclic dependency).
/// This has to be cleaned up, when HLFIR is the default.
bool mayHaveAllocatableComponent(mlir::Type ty);

} // namespace hlfir

#endif // FORTRAN_OPTIMIZER_HLFIR_HLFIRDIALECT_H
