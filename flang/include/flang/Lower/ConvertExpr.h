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

#include "flang/Evaluate/shape.h"
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

namespace Fortran::lower {

class AbstractConverter;
class MaskExpr;
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

/// Create the address of the box.
/// \p expr must be the designator of an allocatable/pointer entity.
fir::MutableBoxValue
createSomeMutableBox(mlir::Location loc, AbstractConverter &converter,
                     const evaluate::Expr<evaluate::SomeType> &expr,
                     SymMap &symMap);

/// Lower an array assignment expression.
///
/// 1. Evaluate the lhs to determine the rank and how to form the ArrayLoad
/// (e.g., if there is a slicing op).
/// 2. Scan the rhs, creating the ArrayLoads and evaluate the scalar subparts to
/// be added to the map.
/// 3. Create the loop nest and evaluate the elemental expression, threading the
/// results.
/// 4. Copy the resulting array back with ArrayMergeStore to the lhs as
/// determined per step 1.
void createSomeArrayAssignment(AbstractConverter &converter,
                               const evaluate::Expr<evaluate::SomeType> &lhs,
                               const evaluate::Expr<evaluate::SomeType> &rhs,
                               SymMap &symMap, StatementContext &stmtCtx);

/// Lower an array assignment expression with masking expression(s).
///
/// 1. Evaluate the lhs to determine the rank and how to form the ArrayLoad
/// (e.g., if there is a slicing op).
/// 2. Scan the rhs, creating the ArrayLoads and evaluate the scalar subparts to
/// be added to the map.
/// 3. Create the loop nest.
/// 4. Create the masking condition. Step 5 is conditionally executed only when
/// the mask condition evaluates to true.
/// 5. Evaluate the elemental expression, threading the results.
/// 6. Copy the resulting array back with ArrayMergeStore to the lhs as
/// determined per step 1.
void createMaskedArrayAssignment(AbstractConverter &converter,
                                 const evaluate::Expr<evaluate::SomeType> &lhs,
                                 const evaluate::Expr<evaluate::SomeType> &rhs,
                                 Fortran::lower::MaskExpr &masks,
                                 SymMap &symMap, StatementContext &stmtCtx);

/// Create an array temporary.
/// When lowering an array expression, it may be necessary to allocate temporary
/// space for a ephemeral array value to be stored.
fir::AllocMemOp
createSomeArrayTemp(AbstractConverter &converter,
                    const evaluate::Expr<evaluate::SomeType> &expr,
                    SymMap &symMap, StatementContext &stmtCtx);

/// Lower an array expression with "parallel" semantics. Such a rhs expression
/// is fully evaluated prior to being assigned back to the destination array.
fir::ExtendedValue
createSomeNewArrayValue(AbstractConverter &converter, fir::ArrayLoadOp dst,
                        const std::optional<evaluate::Shape> &shape,
                        const evaluate::Expr<evaluate::SomeType> &expr,
                        SymMap &symMap, StatementContext &stmtCtx);

/// Lower an array expression to a value of type box.
fir::ExtendedValue
createSomeArrayBox(AbstractConverter &converter,
                   const evaluate::Expr<evaluate::SomeType> &expr,
                   SymMap &symMap, StatementContext &stmtCtx);

} // namespace Fortran::lower

#endif // FORTRAN_LOWER_CONVERTEXPR_H
