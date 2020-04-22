//===-- Lower/ConvertExpr.h -- lowering of expressions ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_CONVERT_EXPR_H
#define FORTRAN_LOWER_CONVERT_EXPR_H

/// [Coding style](https://llvm.org/docs/CodingStandards.html)

namespace mlir {
class Location;
class OpBuilder;
class Type;
class Value;
} // namespace mlir

namespace fir {
class AllocaExpr;
} // namespace fir

namespace Fortran {
namespace common {
class IntrinsicTypeDefaultKinds;
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
class FirOpBuilder;
class SymMap;

/// Create an expression.
/// Lowers `expr` to the FIR dialect of MLIR. The expression is lowered to a
/// value result.
mlir::Value createSomeExpression(mlir::Location loc,
                                 AbstractConverter &converter,
                                 const evaluate::Expr<evaluate::SomeType> &expr,
                                 SymMap &symMap);

/// Create an address.
/// Lowers `expr` to the FIR dialect of MLIR. The expression must be an entity
/// and the address of the entity is returned.
mlir::Value createSomeAddress(mlir::Location loc, AbstractConverter &converter,
                              const evaluate::Expr<evaluate::SomeType> &expr,
                              SymMap &symMap);

} // namespace lower
} // namespace Fortran

#endif // FORTRAN_LOWER_CONVERT_EXPR_H
