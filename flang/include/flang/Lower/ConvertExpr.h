//===-- Lower/ConvertExpr.h -- lowering of expressions ----------*- C++ -*-===//
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
/// Implements the conversion from Fortran::evaluate::Expr trees to FIR.
///
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_CONVERTEXPR_H
#define FORTRAN_LOWER_CONVERTEXPR_H

#include "flang/Lower/Support/BoxValue.h"

namespace mlir {
class Location;
class Value;
} // namespace mlir

namespace fir {
class AllocMemOp;
class ArrayLoadOp;
class ShapeOp;
} // namespace fir

namespace Fortran {
namespace evaluate {
template <typename>
class Expr;
struct SomeType;
} // namespace evaluate

namespace lower {

class AbstractConverter;
class StatementContext;
class SymMap;

/// Create an extended expression value.
fir::ExtendedValue
createSomeExtendedExpression(mlir::Location loc, AbstractConverter &converter,
                             const evaluate::Expr<evaluate::SomeType> &expr,
                             SymMap &symMap, StatementContext &stmtCtx);

fir::ExtendedValue
createSomeInitializerExpression(mlir::Location loc,
                                AbstractConverter &converter,
                                const evaluate::Expr<evaluate::SomeType> &expr,
                                SymMap &symMap, StatementContext &stmtCtx);

/// Create an extended expression address.
fir::ExtendedValue
createSomeExtendedAddress(mlir::Location loc, AbstractConverter &converter,
                          const evaluate::Expr<evaluate::SomeType> &expr,
                          SymMap &symMap, StatementContext &stmtCtx);

/// Create a string literal. Lowers `str` to the MLIR representation of a
/// literal CHARACTER value. (KIND is assumed to be 1.)
fir::ExtendedValue createStringLiteral(mlir::Location loc,
                                       AbstractConverter &converter,
                                       llvm::StringRef str, std::uint64_t len);

/// Create and return a projection of a subspace of an array value. This is the
/// lhs onto which a newly constructed array value can be merged.
fir::ArrayLoadOp
createSomeArraySubspace(AbstractConverter &converter,
                        const evaluate::Expr<evaluate::SomeType> &expr,
                        SymMap &symMap, StatementContext &stmtCtx);

/// Create an array temporary.
fir::AllocMemOp
createSomeArrayTemp(AbstractConverter &converter,
                    const evaluate::Expr<evaluate::SomeType> &expr,
                    SymMap &symMap, StatementContext &stmtCtx);

/// Lower an array expression with "parallel" semantics. Such a rhs expression
/// is fully evaluated prior to being assigned back to the destination array.
fir::ExtendedValue
createSomeNewArrayValue(AbstractConverter &converter, fir::ArrayLoadOp dst,
                        const evaluate::Expr<evaluate::SomeType> &expr,
                        SymMap &symMap, StatementContext &stmtCtx);

} // namespace lower
} // namespace Fortran

#endif // FORTRAN_LOWER_CONVERTEXPR_H
