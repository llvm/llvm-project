//===-- VectorSubscripts.h -- vector subscripts tools -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Defines a compiler internal representation for lowered designators
///  containing vector subscripts. This representation allows working on such
///  designators in custom ways while ensuring the designator subscripts are
///  only evaluated once. It is mainly intended for cases that do not fit in
///  the array expression lowering framework like input IO in presence of
///  vector subscripts.
///
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_VECTORSUBSCRIPTS_H
#define FORTRAN_LOWER_VECTORSUBSCRIPTS_H

#include "flang/Optimizer/Builder/BoxValue.h"

namespace fir {
class FirOpBuilder;
}

namespace Fortran {

namespace evaluate {
template <typename>
class Expr;
struct SomeType;
} // namespace evaluate

namespace lower {

class AbstractConverter;
class StatementContext;

/// VectorSubscriptBox is a lowered representation for any Designator<T> that
/// contain at least one vector subscript.
///
/// A designator `x%a(i,j)%b(1:foo():1, vector, k)%c%d(m)%e1
/// Is lowered into:
///   - an ExtendedValue for ranked base (x%a(i,j)%b)
///   - mlir:Values and ExtendedValues for the triplet, vector subscript and
///     scalar subscripts of the ranked array reference (1:foo():1, vector, k)
///   - a list of fir.field_index and scalar integers mlir::Value for the
///   component
///     path at the right of the ranked array ref (%c%d(m)%e).
///
/// This representation allows later creating loops over the designator elements
/// and fir.array_coor to get the element addresses without re-evaluating any
/// sub-expressions.
class VectorSubscriptBox {
public:
  /// Type of the callbacks that can be passed to work with the element
  /// addresses.
  using ElementalGenerator = std::function<void(const fir::ExtendedValue &)>;
  using ElementalGeneratorWithBoolReturn =
      std::function<mlir::Value(const fir::ExtendedValue &)>;
  struct LoweredVectorSubscript {
    LoweredVectorSubscript(fir::ExtendedValue &&vector, mlir::Value size)
        : vector{std::move(vector)}, size{size} {}
    fir::ExtendedValue vector;
    // Vector size, guaranteed to be of indexType.
    mlir::Value size;
  };
  struct LoweredTriplet {
    // Triplets value, guaranteed to be of indexType.
    mlir::Value lb;
    mlir::Value ub;
    mlir::Value stride;
  };
  using LoweredSubscript =
      std::variant<mlir::Value, LoweredTriplet, LoweredVectorSubscript>;
  using MaybeSubstring = llvm::SmallVector<mlir::Value, 2>;
  VectorSubscriptBox(
      fir::ExtendedValue &&loweredBase,
      llvm::SmallVector<LoweredSubscript, 16> &&loweredSubscripts,
      llvm::SmallVector<mlir::Value> &&componentPath,
      MaybeSubstring substringBounds, mlir::Type elementType)
      : loweredBase{std::move(loweredBase)}, loweredSubscripts{std::move(
                                                 loweredSubscripts)},
        componentPath{std::move(componentPath)},
        substringBounds{substringBounds}, elementType{elementType} {};

  /// Loop over the elements described by the VectorSubscriptBox, and call
  /// \p elementalGenerator inside the loops with the element addresses.
  void loopOverElements(fir::FirOpBuilder &builder, mlir::Location loc,
                        const ElementalGenerator &elementalGenerator);

  /// Loop over the elements described by the VectorSubscriptBox while a
  /// condition is true, and call \p elementalGenerator inside the loops with
  /// the element addresses. The initial condition value is \p initialCondition,
  /// and then it is the result of \p elementalGenerator. The value of the
  /// condition after the loops is returned.
  mlir::Value loopOverElementsWhile(
      fir::FirOpBuilder &builder, mlir::Location loc,
      const ElementalGeneratorWithBoolReturn &elementalGenerator,
      mlir::Value initialCondition);

  /// Return the type of the elements of the array section.
  mlir::Type getElementType() { return elementType; }

private:
  /// Common implementation for DoLoop and IterWhile loop creations.
  template <typename LoopType, typename Generator>
  mlir::Value loopOverElementsBase(fir::FirOpBuilder &builder,
                                   mlir::Location loc,
                                   const Generator &elementalGenerator,
                                   mlir::Value initialCondition);
  /// Create sliceOp for the designator.
  mlir::Value createSlice(fir::FirOpBuilder &builder, mlir::Location loc);

  /// Create ExtendedValue the element inside the loop.
  fir::ExtendedValue getElementAt(fir::FirOpBuilder &builder,
                                  mlir::Location loc, mlir::Value shape,
                                  mlir::Value slice,
                                  mlir::ValueRange inductionVariables);

  /// Generate the [lb, ub, step] to loop over the section (in loop order, not
  /// Fortran dimension order).
  llvm::SmallVector<std::tuple<mlir::Value, mlir::Value, mlir::Value>>
  genLoopBounds(fir::FirOpBuilder &builder, mlir::Location loc);

  /// Lowered base of the ranked array ref.
  fir::ExtendedValue loweredBase;
  /// Subscripts values of the rank arrayRef part.
  llvm::SmallVector<LoweredSubscript, 16> loweredSubscripts;
  /// Scalar subscripts and components at the right of the ranked
  /// array ref part of any.
  llvm::SmallVector<mlir::Value> componentPath;
  /// List of substring bounds if this is a substring (only the lower bound if
  /// the upper is implicit).
  MaybeSubstring substringBounds;
  /// Type of the elements described by this array section.
  mlir::Type elementType;
};

/// Lower \p expr, that must be an designator containing vector subscripts, to a
/// VectorSubscriptBox representation. This causes evaluation of all the
/// subscripts. Any required clean-ups from subscript expression are added to \p
/// stmtCtx.
VectorSubscriptBox genVectorSubscriptBox(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    Fortran::lower::StatementContext &stmtCtx,
    const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &expr);

} // namespace lower
} // namespace Fortran

#endif // FORTRAN_LOWER_VECTORSUBSCRIPTS_H
