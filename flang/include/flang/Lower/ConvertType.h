//===-- Lower/ConvertType.h -- lowering of types ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//----------------------------------------------------------------------------//
///
/// Conversion of front-end TYPE, KIND, ATTRIBUTE (TKA) information to FIR/MLIR.
/// This is meant to be the single point of truth (SPOT) for all type
/// conversions when lowering to FIR.  This implements all lowering of parse
/// tree TKA to the FIR type system. If one is converting front-end types and
/// not using one of the routines provided here, it's being done wrong.
///
/// [Coding style](https://llvm.org/docs/CodingStandards.html)
///
//----------------------------------------------------------------------------//

#ifndef FORTRAN_LOWER_CONVERT_TYPE_H
#define FORTRAN_LOWER_CONVERT_TYPE_H

#include "flang/Common/Fortran.h"
#include "mlir/IR/Types.h"

namespace mlir {
class Location;
class MLIRContext;
class Type;
} // namespace mlir

namespace Fortran {
namespace common {
class IntrinsicTypeDefaultKinds;
template <typename>
class Reference;
} // namespace common

namespace evaluate {
struct DataRef;
template <typename>
class Designator;
template <typename>
class Expr;
template <common::TypeCategory>
struct SomeKind;
struct SomeType;
template <common::TypeCategory, int>
class Type;
} // namespace evaluate

namespace semantics {
class Symbol;
} // namespace semantics

namespace lower {
class AbstractConverter;
namespace pft {
struct Variable;
}

using SomeExpr = evaluate::Expr<evaluate::SomeType>;
using SymbolRef = common::Reference<const semantics::Symbol>;

// Type for compile time constant length type parameters.
using LenParameterTy = std::int64_t;

/// Get a FIR type based on a category and kind.
mlir::Type getFIRType(mlir::MLIRContext *ctxt, common::TypeCategory tc,
                      int kind);

/// Get a FIR type based on a category.
mlir::Type getFIRType(Fortran::lower::AbstractConverter &,
                      common::TypeCategory tc);

/// Translate a Fortran::evaluate::DataRef to an mlir::Type.
mlir::Type translateDataRefToFIRType(Fortran::lower::AbstractConverter &,
                                     const evaluate::DataRef &dataRef);

/// Translate a SomeExpr to an mlir::Type.
mlir::Type translateSomeExprToFIRType(Fortran::lower::AbstractConverter &,
                                      const SomeExpr &expr);

/// Translate a Fortran::semantics::Symbol to an mlir::Type.
mlir::Type translateSymbolToFIRType(Fortran::lower::AbstractConverter &,
                                    const SymbolRef symbol);

/// Translate a Fortran::lower::pft::Variable to an mlir::Type.
mlir::Type translateVariableToFIRType(Fortran::lower::AbstractConverter &,
                                      const pft::Variable &variable);

/// Translate a REAL of KIND to the mlir::Type.
mlir::Type convertReal(mlir::MLIRContext *ctxt, int KIND);

// Given a ReferenceType of a base type, returns the ReferenceType to
// the SequenceType of this base type.
// The created SequenceType has one dimension of unknown extent.
// This is useful to do pointer arithmetic using fir::CoordinateOp that requires
// a memory reference to a sequence type.
mlir::Type getSequenceRefType(mlir::Type referenceType);

} // namespace lower
} // namespace Fortran

#endif // FORTRAN_LOWER_CONVERT_TYPE_H
