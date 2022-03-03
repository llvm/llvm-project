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
}

namespace Fortran::evaluate {
template <typename>
class Expr;
struct SomeType;
} // namespace Fortran::evaluate

namespace Fortran::lower {

class AbstractConverter;
class StatementContext;
class SymMap;
class ExplicitIterSpace;
class ImplicitIterSpace;
class StatementContext;

using SomeExpr = Fortran::evaluate::Expr<Fortran::evaluate::SomeType>;

/// Create an extended expression value.
fir::ExtendedValue createSomeExtendedExpression(mlir::Location loc,
                                                AbstractConverter &converter,
                                                const SomeExpr &expr,
                                                SymMap &symMap,
                                                StatementContext &stmtCtx);

/// Create a global array symbol with the Dense attribute
fir::GlobalOp createDenseGlobal(mlir::Location loc, mlir::Type symTy,
                                llvm::StringRef globalName,
                                mlir::StringAttr linkage, bool isConst,
                                const SomeExpr &expr,
                                Fortran::lower::AbstractConverter &converter);

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
                                 SymMap &symMap, StatementContext &stmtCtx);

/// Create the address of the box.
/// \p expr must be the designator of an allocatable/pointer entity.
fir::MutableBoxValue createMutableBox(mlir::Location loc,
                                      AbstractConverter &converter,
                                      const SomeExpr &expr, SymMap &symMap);

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

/// Lower an array assignment expression with pre-evaluated left and right
/// hand sides. This implements an array copy taking into account
/// non-contiguity and potential overlaps.
void createSomeArrayAssignment(AbstractConverter &converter,
                               const fir::ExtendedValue &lhs,
                               const fir::ExtendedValue &rhs, SymMap &symMap,
                               StatementContext &stmtCtx);

/// Lower an assignment to an allocatable array, allocating the array if
/// it is not allocated yet or reallocation it if it does not conform
/// with the right hand side.
void createAllocatableArrayAssignment(AbstractConverter &converter,
                                      const SomeExpr &lhs, const SomeExpr &rhs,
                                      ExplicitIterSpace &explicitIterSpace,
                                      ImplicitIterSpace &implicitIterSpace,
                                      SymMap &symMap,
                                      StatementContext &stmtCtx);

// Attribute for an alloca that is a trivial adaptor for converting a value to
// pass-by-ref semantics for a VALUE parameter. The optimizer may be able to
// eliminate these.
inline mlir::NamedAttribute getAdaptToByRefAttr(fir::FirOpBuilder &builder) {
  return {mlir::StringAttr::get(builder.getContext(),
                                fir::getAdaptToByRefAttrName()),
          builder.getUnitAttr()};
}

/// Generate max(\p value, 0) where \p value is a scalar integer.
mlir::Value genMaxWithZero(fir::FirOpBuilder &builder, mlir::Location loc,
                           mlir::Value value);

} // namespace Fortran::lower

#endif // FORTRAN_LOWER_CONVERTEXPR_H
