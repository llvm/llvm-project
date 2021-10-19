//===-- include/flang/Evaluate/shape.h --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// GetShape() analyzes an expression and determines its shape, if possible,
// representing the result as a vector of scalar integer expressions.

#ifndef FORTRAN_EVALUATE_SHAPE_H_
#define FORTRAN_EVALUATE_SHAPE_H_

#include "expression.h"
#include "fold.h"
#include "traverse.h"
#include "variable.h"
#include "flang/Common/indirection.h"
#include "flang/Evaluate/tools.h"
#include "flang/Evaluate/type.h"
#include <optional>
#include <variant>

namespace Fortran::parser {
class ContextualMessages;
}

namespace Fortran::evaluate {

class FoldingContext;

using ExtentType = SubscriptInteger;
using ExtentExpr = Expr<ExtentType>;
using MaybeExtentExpr = std::optional<ExtentExpr>;
using Shape = std::vector<MaybeExtentExpr>;

bool IsImpliedShape(const Symbol &);
bool IsExplicitShape(const Symbol &);

// Conversions between various representations of shapes.
std::optional<ExtentExpr> AsExtentArrayExpr(const Shape &);

std::optional<Constant<ExtentType>> AsConstantShape(
    FoldingContext &, const Shape &);
Constant<ExtentType> AsConstantShape(const ConstantSubscripts &);

ConstantSubscripts AsConstantExtents(const Constant<ExtentType> &);
std::optional<ConstantSubscripts> AsConstantExtents(
    FoldingContext &, const Shape &);
Shape AsShape(const ConstantSubscripts &);
std::optional<Shape> AsShape(const std::optional<ConstantSubscripts> &);

inline int GetRank(const Shape &s) { return static_cast<int>(s.size()); }

Shape Fold(FoldingContext &, Shape &&);
std::optional<Shape> Fold(FoldingContext &, std::optional<Shape> &&);

template <typename A>
std::optional<Shape> GetShape(FoldingContext &, const A &);
template <typename A> std::optional<Shape> GetShape(const A &);

// The dimension argument to these inquiries is zero-based,
// unlike the DIM= arguments to many intrinsics.
ExtentExpr GetLowerBound(const NamedEntity &, int dimension);
ExtentExpr GetLowerBound(FoldingContext &, const NamedEntity &, int dimension);
MaybeExtentExpr GetUpperBound(const NamedEntity &, int dimension);
MaybeExtentExpr GetUpperBound(
    FoldingContext &, const NamedEntity &, int dimension);
MaybeExtentExpr ComputeUpperBound(ExtentExpr &&lower, MaybeExtentExpr &&extent);
MaybeExtentExpr ComputeUpperBound(
    FoldingContext &, ExtentExpr &&lower, MaybeExtentExpr &&extent);
Shape GetLowerBounds(const NamedEntity &);
Shape GetLowerBounds(FoldingContext &, const NamedEntity &);
Shape GetUpperBounds(const NamedEntity &);
Shape GetUpperBounds(FoldingContext &, const NamedEntity &);
MaybeExtentExpr GetExtent(const NamedEntity &, int dimension);
MaybeExtentExpr GetExtent(FoldingContext &, const NamedEntity &, int dimension);
MaybeExtentExpr GetExtent(
    const Subscript &, const NamedEntity &, int dimension);
MaybeExtentExpr GetExtent(
    FoldingContext &, const Subscript &, const NamedEntity &, int dimension);

// Compute an element count for a triplet or trip count for a DO.
ExtentExpr CountTrips(
    ExtentExpr &&lower, ExtentExpr &&upper, ExtentExpr &&stride);
ExtentExpr CountTrips(
    const ExtentExpr &lower, const ExtentExpr &upper, const ExtentExpr &stride);
MaybeExtentExpr CountTrips(
    MaybeExtentExpr &&lower, MaybeExtentExpr &&upper, MaybeExtentExpr &&stride);

// Computes SIZE() == PRODUCT(shape)
MaybeExtentExpr GetSize(Shape &&);
ConstantSubscript GetSize(const ConstantSubscripts &);

// Utility predicate: does an expression reference any implied DO index?
bool ContainsAnyImpliedDoIndex(const ExtentExpr &);

class GetShapeHelper
    : public AnyTraverse<GetShapeHelper, std::optional<Shape>> {
public:
  using Result = std::optional<Shape>;
  using Base = AnyTraverse<GetShapeHelper, Result>;
  using Base::operator();
  GetShapeHelper() : Base{*this} {}
  explicit GetShapeHelper(FoldingContext &c) : Base{*this}, context_{&c} {}

  Result operator()(const ImpliedDoIndex &) const { return ScalarShape(); }
  Result operator()(const DescriptorInquiry &) const { return ScalarShape(); }
  Result operator()(const TypeParamInquiry &) const { return ScalarShape(); }
  Result operator()(const BOZLiteralConstant &) const { return ScalarShape(); }
  Result operator()(const StaticDataObject::Pointer &) const {
    return ScalarShape();
  }
  Result operator()(const StructureConstructor &) const {
    return ScalarShape();
  }

  template <typename T> Result operator()(const Constant<T> &c) const {
    return ConstantShape(c.SHAPE());
  }

  Result operator()(const Symbol &) const;
  Result operator()(const Component &) const;
  Result operator()(const ArrayRef &) const;
  Result operator()(const CoarrayRef &) const;
  Result operator()(const Substring &) const;
  Result operator()(const ProcedureRef &) const;

  template <typename T>
  Result operator()(const ArrayConstructor<T> &aconst) const {
    return Shape{GetArrayConstructorExtent(aconst)};
  }
  template <typename D, typename R, typename LO, typename RO>
  Result operator()(const Operation<D, R, LO, RO> &operation) const {
    if (operation.right().Rank() > 0) {
      return (*this)(operation.right());
    } else {
      return (*this)(operation.left());
    }
  }

private:
  static Result ScalarShape() { return Shape{}; }
  static Shape ConstantShape(const Constant<ExtentType> &);
  Result AsShape(ExtentExpr &&) const;
  static Shape CreateShape(int rank, NamedEntity &);

  template <typename T>
  MaybeExtentExpr GetArrayConstructorValueExtent(
      const ArrayConstructorValue<T> &value) const {
    return std::visit(
        common::visitors{
            [&](const Expr<T> &x) -> MaybeExtentExpr {
              if (auto xShape{
                      context_ ? GetShape(*context_, x) : GetShape(x)}) {
                // Array values in array constructors get linearized.
                return GetSize(std::move(*xShape));
              } else {
                return std::nullopt;
              }
            },
            [&](const ImpliedDo<T> &ido) -> MaybeExtentExpr {
              // Don't be heroic and try to figure out triangular implied DO
              // nests.
              if (!ContainsAnyImpliedDoIndex(ido.lower()) &&
                  !ContainsAnyImpliedDoIndex(ido.upper()) &&
                  !ContainsAnyImpliedDoIndex(ido.stride())) {
                if (auto nValues{GetArrayConstructorExtent(ido.values())}) {
                  return std::move(*nValues) *
                      CountTrips(ido.lower(), ido.upper(), ido.stride());
                }
              }
              return std::nullopt;
            },
        },
        value.u);
  }

  template <typename T>
  MaybeExtentExpr GetArrayConstructorExtent(
      const ArrayConstructorValues<T> &values) const {
    ExtentExpr result{0};
    for (const auto &value : values) {
      if (MaybeExtentExpr n{GetArrayConstructorValueExtent(value)}) {
        result = std::move(result) + std::move(*n);
        if (context_) {
          // Fold during expression creation to avoid creating an expression so
          // large we can't evalute it without overflowing the stack.
          result = Fold(*context_, std::move(result));
        }
      } else {
        return std::nullopt;
      }
    }
    return result;
  }

  FoldingContext *context_{nullptr};
};

template <typename A>
std::optional<Shape> GetShape(FoldingContext &context, const A &x) {
  if (auto shape{GetShapeHelper{context}(x)}) {
    return Fold(context, std::move(shape));
  } else {
    return std::nullopt;
  }
}

template <typename A> std::optional<Shape> GetShape(const A &x) {
  return GetShapeHelper{}(x);
}

template <typename A>
std::optional<Shape> GetShape(FoldingContext *context, const A &x) {
  if (context) {
    return GetShape(*context, x);
  } else {
    return GetShapeHelper{}(x);
  }
}

template <typename A>
std::optional<Constant<ExtentType>> GetConstantShape(
    FoldingContext &context, const A &x) {
  if (auto shape{GetShape(context, x)}) {
    return AsConstantShape(context, *shape);
  } else {
    return std::nullopt;
  }
}

template <typename A>
std::optional<ConstantSubscripts> GetConstantExtents(
    FoldingContext &context, const A &x) {
  if (auto shape{GetShape(context, x)}) {
    return AsConstantExtents(context, *shape);
  } else {
    return std::nullopt;
  }
}

// Compilation-time shape conformance checking, when corresponding extents
// are or should be known.  The result is an optional Boolean:
//  - nullopt: no error found or reported, but conformance cannot
//    be guaranteed during compilation; this result is possible only
//    when one or both arrays are allowed to have deferred shape
//  - true: no error found or reported, arrays conform
//  - false: errors found and reported
// Use "CheckConformance(...).value_or()" to specify a default result
// when you don't care whether messages have been emitted.
struct CheckConformanceFlags {
  enum Flags {
    None = 0,
    LeftScalarExpandable = 1,
    RightScalarExpandable = 2,
    LeftIsDeferredShape = 4,
    RightIsDeferredShape = 8,
    EitherScalarExpandable = LeftScalarExpandable | RightScalarExpandable,
    BothDeferredShape = LeftIsDeferredShape | RightIsDeferredShape,
    RightIsExpandableDeferred = RightScalarExpandable | RightIsDeferredShape,
  };
};
std::optional<bool> CheckConformance(parser::ContextualMessages &,
    const Shape &left, const Shape &right,
    CheckConformanceFlags::Flags flags = CheckConformanceFlags::None,
    const char *leftIs = "left operand", const char *rightIs = "right operand");

// Increments one-based subscripts in element order (first varies fastest)
// and returns true when they remain in range; resets them all to one and
// return false otherwise (including the case where one or more of the
// extents are zero).
bool IncrementSubscripts(
    ConstantSubscripts &, const ConstantSubscripts &extents);

} // namespace Fortran::evaluate
#endif // FORTRAN_EVALUATE_SHAPE_H_
