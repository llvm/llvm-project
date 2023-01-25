//===-- Lower/ConvertExprToHLFIR.h -- lowering of expressions ----*- C++-*-===//
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
/// Implements the conversion from Fortran::evaluate::Expr trees to HLFIR.
///
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_CONVERTEXPRTOHLFIR_H
#define FORTRAN_LOWER_CONVERTEXPRTOHLFIR_H

#include "flang/Lower/StatementContext.h"
#include "flang/Lower/Support/Utils.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/HLFIRTools.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"

namespace mlir {
class Location;
} // namespace mlir

namespace Fortran::lower {

class AbstractConverter;
class SymMap;

hlfir::EntityWithAttributes
convertExprToHLFIR(mlir::Location loc, Fortran::lower::AbstractConverter &,
                   const Fortran::lower::SomeExpr &, Fortran::lower::SymMap &,
                   Fortran::lower::StatementContext &);

inline fir::ExtendedValue
translateToExtendedValue(mlir::Location loc, fir::FirOpBuilder &builder,
                         hlfir::Entity entity,
                         Fortran::lower::StatementContext &context) {
  auto [exv, exvCleanup] =
      hlfir::translateToExtendedValue(loc, builder, entity);
  if (exvCleanup)
    context.attachCleanup(*exvCleanup);
  return exv;
}

/// Lower an evaluate::Expr to a fir::Box.
fir::BoxValue convertExprToBox(mlir::Location loc,
                               Fortran::lower::AbstractConverter &,
                               const Fortran::lower::SomeExpr &,
                               Fortran::lower::SymMap &,
                               Fortran::lower::StatementContext &);
fir::BoxValue convertToBox(mlir::Location loc,
                           Fortran::lower::AbstractConverter &,
                           hlfir::Entity entity,
                           Fortran::lower::StatementContext &);

/// Lower an evaluate::Expr to fir::ExtendedValue address.
/// The address may be a raw fir.ref<T>, or a fir.box<T>/fir.class<T>, (pointer
/// and allocatable are dereferenced).
/// - If expression is not a variable, or is a designator with vector
///   subscripts, a temporary is created to hold the expression value and
///   is returned as:
///   - a fir.class<T> if the expression is polymorphic.
///   - otherwise, a fir.box<T> if it is a derived type with length
///     parameters (not yet implemented).
///   - otherwise, a fir.ref<T>
/// - If the expression is a variable that is not a designator with
///   vector subscripts, it is lowered without creating a temporary and
///   is returned as:
///   - a fir.class<T> if the variable is polymorphic.
///   - otherwise, a fir.box<T> if it is a derived type with length
///     parameters (not yet implemented), or if it is not a simply
///     contiguous.
///   - otherwise, a fir.ref<T>
///
/// Beware that this is different from the previous createSomeExtendedAddress
/// that had a non-trivial behaviour and would create contiguous temporary for
/// array sections `x(:, :)`, but not for `x` even if x is not simply
/// contiguous.
fir::ExtendedValue convertExprToAddress(mlir::Location loc,
                                        Fortran::lower::AbstractConverter &,
                                        const Fortran::lower::SomeExpr &,
                                        Fortran::lower::SymMap &,
                                        Fortran::lower::StatementContext &);
fir::ExtendedValue convertToAddress(mlir::Location loc,
                                    Fortran::lower::AbstractConverter &,
                                    hlfir::Entity entity,
                                    Fortran::lower::StatementContext &);

/// Lower an evaluate::Expr to a fir::ExtendedValue value.
fir::ExtendedValue convertExprToValue(mlir::Location loc,
                                      Fortran::lower::AbstractConverter &,
                                      const Fortran::lower::SomeExpr &,
                                      Fortran::lower::SymMap &,
                                      Fortran::lower::StatementContext &);
fir::ExtendedValue convertToValue(mlir::Location loc,
                                  Fortran::lower::AbstractConverter &,
                                  hlfir::Entity entity,
                                  Fortran::lower::StatementContext &);

/// Lower an evaluate::Expr to a fir::MutableBoxValue value.
/// This can only be called if the Expr is a POINTER or ALLOCATABLE,
/// otherwise, this will crash.
fir::MutableBoxValue
convertExprToMutableBox(mlir::Location loc, Fortran::lower::AbstractConverter &,
                        const Fortran::lower::SomeExpr &,
                        Fortran::lower::SymMap &);
} // namespace Fortran::lower

#endif // FORTRAN_LOWER_CONVERTEXPRTOHLFIR_H
