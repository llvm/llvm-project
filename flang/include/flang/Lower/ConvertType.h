//===-- Lower/ConvertType.h -- lowering of types ----------------*- C++ -*-===//
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
///
/// Conversion of front-end TYPE, KIND, ATTRIBUTE (TKA) information to FIR/MLIR.
/// This is meant to be the single point of truth (SPOT) for all type
/// conversions when lowering to FIR.  This implements all lowering of parse
/// tree TKA to the FIR type system. If one is converting front-end types and
/// not using one of the routines provided here, it's being done wrong.
///
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_CONVERT_TYPE_H
#define FORTRAN_LOWER_CONVERT_TYPE_H

#include "flang/Common/Fortran.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
class Location;
class MLIRContext;
class Type;
} // namespace mlir

namespace Fortran {
namespace common {
template <typename>
class Reference;
} // namespace common

namespace evaluate {
template <typename>
class Expr;
struct SomeType;
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
                      int kind, llvm::ArrayRef<LenParameterTy>);

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

} // namespace lower
} // namespace Fortran

#endif // FORTRAN_LOWER_CONVERT_TYPE_H
