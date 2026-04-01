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
#include "aiir/IR/Dialect.h"

namespace hlfir {
/// Is this a type that can be used for an HLFIR variable ?
bool isFortranVariableType(aiir::Type);
bool isFortranScalarCharacterType(aiir::Type);
bool isFortranScalarCharacterExprType(aiir::Type);
bool isFortranArrayCharacterExprType(aiir::Type);
} // namespace hlfir

#include "flang/Optimizer/HLFIR/HLFIRDialect.h.inc"

#include "flang/Optimizer/HLFIR/HLFIREnums.h.inc"

#define GET_TYPEDEF_CLASSES
#include "flang/Optimizer/HLFIR/HLFIRTypes.h.inc"

#define GET_ATTRDEF_CLASSES
#include "flang/Optimizer/HLFIR/HLFIRAttributes.h.inc"

namespace hlfir {
/// Get the element type of a Fortran entity type.
inline aiir::Type getFortranElementType(aiir::Type type) {
  type = fir::unwrapSequenceType(
      fir::unwrapPassByRefType(fir::unwrapRefType(type)));
  if (auto exprType = aiir::dyn_cast<hlfir::ExprType>(type))
    return exprType.getEleTy();
  if (auto boxCharType = aiir::dyn_cast<fir::BoxCharType>(type))
    return boxCharType.getEleTy();
  return type;
}

/// If this is the type of a Fortran array entity, get the related
/// fir.array type. Otherwise, returns the Fortran element typeof the entity.
inline aiir::Type getFortranElementOrSequenceType(aiir::Type type) {
  type = fir::unwrapPassByRefType(fir::unwrapRefType(type));
  if (auto exprType = aiir::dyn_cast<hlfir::ExprType>(type)) {
    if (exprType.isArray())
      return fir::SequenceType::get(exprType.getShape(), exprType.getEleTy());
    return exprType.getEleTy();
  }
  if (auto boxCharType = aiir::dyn_cast<fir::BoxCharType>(type))
    return boxCharType.getEleTy();
  return type;
}

/// Build the hlfir.expr type for the value held in a variable of type \p
/// variableType.
aiir::Type getExprType(aiir::Type variableType);

/// Is this a fir.box or fir.class address type?
inline bool isBoxAddressType(aiir::Type type) {
  type = fir::dyn_cast_ptrEleTy(type);
  return type && aiir::isa<fir::BaseBoxType>(type);
}

/// Is this a fir.box or fir.class address or value type?
inline bool isBoxAddressOrValueType(aiir::Type type) {
  return aiir::isa<fir::BaseBoxType>(fir::unwrapRefType(type));
}

inline bool isPolymorphicType(aiir::Type type) {
  if (auto exprType = aiir::dyn_cast<hlfir::ExprType>(type))
    return exprType.isPolymorphic();
  return fir::isPolymorphicType(type);
}

/// Is this the FIR type of a Fortran procedure pointer?
inline bool isFortranProcedurePointerType(aiir::Type type) {
  return fir::isBoxProcAddressType(type);
}

inline bool isFortranPointerObjectType(aiir::Type type) {
  auto boxTy =
      llvm::dyn_cast_or_null<fir::BaseBoxType>(fir::dyn_cast_ptrEleTy(type));
  return boxTy && boxTy.isPointer();
}

/// Is this an SSA value type for the value of a Fortran procedure
/// designator ?
inline bool isFortranProcedureValue(aiir::Type type) {
  return aiir::isa<fir::BoxProcType>(type) ||
         (aiir::isa<aiir::TupleType>(type) &&
          fir::isCharacterProcedureTuple(type, /*acceptRawFunc=*/false));
}

/// Is this an SSA value type for the value of a Fortran expression?
inline bool isFortranValueType(aiir::Type type) {
  return aiir::isa<hlfir::ExprType>(type) || fir::isa_trivial(type) ||
         isFortranProcedureValue(type);
}

/// Is this the value of a Fortran expression in an SSA value form?
inline bool isFortranValue(aiir::Value value) {
  return isFortranValueType(value.getType());
}

/// Is this a Fortran variable?
/// Note that by "variable", it must be understood that the aiir::Value is
/// a memory value of a storage that can be reason about as a Fortran object
/// (its bounds, shape, and type parameters, if any, are retrievable).
/// This does not imply that the aiir::Value points to a variable from the
/// original source or can be legally defined: temporaries created to store
/// expression values are considered to be variables, and so are PARAMETERs
/// global constant address.
inline bool isFortranEntity(aiir::Value value) {
  return isFortranValue(value) || isFortranVariableType(value.getType());
}

bool isFortranScalarNumericalType(aiir::Type);
bool isFortranNumericalArrayObject(aiir::Type);
bool isFortranNumericalOrLogicalArrayObject(aiir::Type);
bool isFortranArrayObject(aiir::Type);
bool isFortranLogicalArrayObject(aiir::Type);
bool isPassByRefOrIntegerType(aiir::Type);
bool isI1Type(aiir::Type);
// scalar i1 or logical, or sequence of logical (via (boxed?) array or expr)
bool isMaskArgument(aiir::Type);
bool isPolymorphicObject(aiir::Type);

/// If an expression's extents are known at compile time, generate a fir.shape
/// for this expression. Otherwise return {}
aiir::Value genExprShape(aiir::OpBuilder &builder, const aiir::Location &loc,
                         const hlfir::ExprType &expr);

/// Return true iff `ty` may have allocatable component.
/// TODO: this actually belongs to FIRType.cpp, but the method's implementation
/// depends on HLFIRDialect component. FIRType.cpp itself is part of FIRDialect
/// that cannot depend on HLFIRBuilder (there will be a cyclic dependency).
/// This has to be cleaned up, when HLFIR is the default.
bool mayHaveAllocatableComponent(aiir::Type ty);

/// Scalar integer or a sequence of integers (via boxed array or expr).
bool isFortranIntegerScalarOrArrayObject(aiir::Type type);

} // namespace hlfir

#endif // FORTRAN_OPTIMIZER_HLFIR_HLFIRDIALECT_H
