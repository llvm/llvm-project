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

#include "flang/Lower/Support/Utils.h"
#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"

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
class ExplicitIterSpace;
class ImplicitIterSpace;
class StatementContext;
class SymMap;

/// Create an extended expression value.
fir::ExtendedValue createSomeExtendedExpression(mlir::Location loc,
                                                AbstractConverter &converter,
                                                const SomeExpr &expr,
                                                SymMap &symMap,
                                                StatementContext &stmtCtx);

/// Create the IR for the expression \p expr in an initialization context.
/// Expressions that appear in initializers may not allocate temporaries, do not
/// have a stack, etc.
fir::ExtendedValue createSomeInitializerExpression(mlir::Location loc,
                                                   AbstractConverter &converter,
                                                   const SomeExpr &expr,
                                                   SymMap &symMap,
                                                   StatementContext &stmtCtx);

/// Create an extended expression address.
fir::ExtendedValue createSomeExtendedAddress(mlir::Location loc,
                                             AbstractConverter &converter,
                                             const SomeExpr &expr,
                                             SymMap &symMap,
                                             StatementContext &stmtCtx);

/// Create an address in an initializer context. Must be a constant or a symbol
/// to be resolved at link-time. Expressions that appear in initializers may not
/// allocate temporaries, do not have a stack, etc.
fir::ExtendedValue createInitializerAddress(mlir::Location loc,
                                            AbstractConverter &converter,
                                            const SomeExpr &expr,
                                            SymMap &symMap,
                                            StatementContext &stmtCtx);

/// Create the address of the box.
/// \p expr must be the designator of an allocatable/pointer entity.
fir::MutableBoxValue createMutableBox(mlir::Location loc,
                                      AbstractConverter &converter,
                                      const SomeExpr &expr, SymMap &symMap);

/// Create a fir::BoxValue describing the value of \p expr.
/// If \p expr is a variable without vector subscripts, the fir::BoxValue
/// described the variable storage. Otherwise, the created fir::BoxValue
/// describes a temporary storage containing \p expr evaluation, and clean-up
/// for the temporary is added to the provided StatementContext \p stmtCtx.
fir::ExtendedValue createBoxValue(mlir::Location loc,
                                  AbstractConverter &converter,
                                  const SomeExpr &expr, SymMap &symMap,
                                  StatementContext &stmtCtx);

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
                               const SomeExpr &lhs, const SomeExpr &rhs,
                               SymMap &symMap, StatementContext &stmtCtx);

/// Lower an array assignment expression with a pre-evaluated left hand side.
///
/// 1. Scan the rhs, creating the ArrayLoads and evaluate the scalar subparts to
/// be added to the map.
/// 2. Create the loop nest and evaluate the elemental expression, threading the
/// results.
/// 3. Copy the resulting array back with ArrayMergeStore to the lhs as
/// determined per step 1.
void createSomeArrayAssignment(AbstractConverter &converter,
                               const fir::ExtendedValue &lhs,
                               const SomeExpr &rhs, SymMap &symMap,
                               StatementContext &stmtCtx);

/// Lower an array assignment expression with pre-evaluated left and right
/// hand sides. This implements an array copy taking into account
/// non-contiguity and potential overlaps.
void createSomeArrayAssignment(AbstractConverter &converter,
                               const fir::ExtendedValue &lhs,
                               const fir::ExtendedValue &rhs, SymMap &symMap,
                               StatementContext &stmtCtx);

/// Common entry point for both explicit iteration spaces and implicit iteration
/// spaces with masks.
///
/// For an implicit iteration space with masking, lowers an array assignment
/// expression with masking expression(s).
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
///
/// For an explicit iteration space, lower a scalar or array assignment
/// expression with a user-defined iteration space and possibly with masking
/// expression(s).
///
/// If the expression is scalar, then the assignment is an array assignment but
/// the array accesses are explicitly defined by the user and not implied for
/// each element in the array. Mask expressions are optional.
///
/// If the expression has rank, then the assignment has a combined user-defined
/// iteration space as well as a inner (subordinate) implied iteration
/// space. The implied iteration space may include WHERE conditions, `masks`.
void createAnyMaskedArrayAssignment(AbstractConverter &converter,
                                    const SomeExpr &lhs, const SomeExpr &rhs,
                                    ExplicitIterSpace &explicitIterSpace,
                                    ImplicitIterSpace &implicitIterSpace,
                                    SymMap &symMap, StatementContext &stmtCtx);

/// Lower an assignment to an allocatable array, allocating the array if
/// it is not allocated yet or reallocation it if it does not conform
/// with the right hand side.
void createAllocatableArrayAssignment(AbstractConverter &converter,
                                      const SomeExpr &lhs, const SomeExpr &rhs,
                                      ExplicitIterSpace &explicitIterSpace,
                                      ImplicitIterSpace &implicitIterSpace,
                                      SymMap &symMap,
                                      StatementContext &stmtCtx);

/// Lower a pointer assignment in an explicit iteration space. The explicit
/// space iterates over a data structure with a type of `!fir.array<...
/// !fir.box<!fir.ptr<T>> ...>`. Lower the assignment by copying the rhs box
/// value to each array element.
void createArrayOfPointerAssignment(
    AbstractConverter &converter, const SomeExpr &lhs, const SomeExpr &rhs,
    ExplicitIterSpace &explicitIterSpace, ImplicitIterSpace &implicitIterSpace,
    const llvm::SmallVector<mlir::Value> &lbounds,
    llvm::Optional<llvm::SmallVector<mlir::Value>> ubounds, SymMap &symMap,
    StatementContext &stmtCtx);

/// Lower an array expression with "parallel" semantics. Such a rhs expression
/// is fully evaluated prior to being assigned back to a temporary array.
fir::ExtendedValue createSomeArrayTempValue(AbstractConverter &converter,
                                            const SomeExpr &expr,
                                            SymMap &symMap,
                                            StatementContext &stmtCtx);

/// Somewhat similar to createSomeArrayTempValue, but the temporary buffer is
/// allocated lazily (inside the loops instead of before the loops) to
/// accomodate buffers with shapes that cannot be precomputed. In fact, the
/// buffer need not even be hyperrectangular. The buffer may be created as an
/// instance of a ragged array, which may be useful if an array's extents are
/// functions of other loop indices. The ragged array structure is built with \p
/// raggedHeader being the root header variable. The header is a tuple of
/// `{rank, data-is-headers, [data]*, [extents]*}`, which is built recursively.
/// The base header, \p raggedHeader, must be initialized to zeros.
void createLazyArrayTempValue(AbstractConverter &converter,
                              const SomeExpr &expr, mlir::Value raggedHeader,
                              SymMap &symMap, StatementContext &stmtCtx);

/// Lower an array expression to a value of type box. The expression must be a
/// variable.
fir::ExtendedValue createSomeArrayBox(AbstractConverter &converter,
                                      const SomeExpr &expr, SymMap &symMap,
                                      StatementContext &stmtCtx);

/// Lower a subroutine call. This handles both elemental and non elemental
/// subroutines. \p isUserDefAssignment must be set if this is called in the
/// context of a user defined assignment. For subroutines with alternate
/// returns, the returned value indicates which label the code should jump to.
/// The returned value is null otherwise.
mlir::Value createSubroutineCall(AbstractConverter &converter,
                                 const evaluate::ProcedureRef &call,
                                 ExplicitIterSpace &explicitIterSpace,
                                 ImplicitIterSpace &implicitIterSpace,
                                 SymMap &symMap, StatementContext &stmtCtx,
                                 bool isUserDefAssignment);

// Attribute for an alloca that is a trivial adaptor for converting a value to
// pass-by-ref semantics for a VALUE parameter. The optimizer may be able to
// eliminate these.
inline mlir::NamedAttribute getAdaptToByRefAttr(fir::FirOpBuilder &builder) {
  return {mlir::StringAttr::get(builder.getContext(),
                                fir::getAdaptToByRefAttrName()),
          builder.getUnitAttr()};
}

} // namespace Fortran::lower

#endif // FORTRAN_LOWER_CONVERTEXPR_H
