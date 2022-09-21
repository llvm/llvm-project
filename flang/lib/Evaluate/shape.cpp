//===-- lib/Evaluate/shape.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Evaluate/shape.h"
#include "flang/Common/idioms.h"
#include "flang/Common/template.h"
#include "flang/Evaluate/characteristics.h"
#include "flang/Evaluate/check-expression.h"
#include "flang/Evaluate/fold.h"
#include "flang/Evaluate/intrinsics.h"
#include "flang/Evaluate/tools.h"
#include "flang/Evaluate/type.h"
#include "flang/Parser/message.h"
#include "flang/Semantics/symbol.h"
#include <functional>

using namespace std::placeholders; // _1, _2, &c. for std::bind()

namespace Fortran::evaluate {

bool IsImpliedShape(const Symbol &original) {
  const Symbol &symbol{ResolveAssociations(original)};
  const auto *details{symbol.detailsIf<semantics::ObjectEntityDetails>()};
  return details && symbol.attrs().test(semantics::Attr::PARAMETER) &&
      details->shape().CanBeImpliedShape();
}

bool IsExplicitShape(const Symbol &original) {
  const Symbol &symbol{ResolveAssociations(original)};
  if (const auto *details{symbol.detailsIf<semantics::ObjectEntityDetails>()}) {
    const auto &shape{details->shape()};
    return shape.Rank() == 0 ||
        shape.IsExplicitShape(); // true when scalar, too
  } else {
    return symbol
        .has<semantics::AssocEntityDetails>(); // exprs have explicit shape
  }
}

Shape GetShapeHelper::ConstantShape(const Constant<ExtentType> &arrayConstant) {
  CHECK(arrayConstant.Rank() == 1);
  Shape result;
  std::size_t dimensions{arrayConstant.size()};
  for (std::size_t j{0}; j < dimensions; ++j) {
    Scalar<ExtentType> extent{arrayConstant.values().at(j)};
    result.emplace_back(MaybeExtentExpr{ExtentExpr{std::move(extent)}});
  }
  return result;
}

auto GetShapeHelper::AsShapeResult(ExtentExpr &&arrayExpr) const -> Result {
  if (context_) {
    arrayExpr = Fold(*context_, std::move(arrayExpr));
  }
  if (const auto *constArray{UnwrapConstantValue<ExtentType>(arrayExpr)}) {
    return ConstantShape(*constArray);
  }
  if (auto *constructor{UnwrapExpr<ArrayConstructor<ExtentType>>(arrayExpr)}) {
    Shape result;
    for (auto &value : *constructor) {
      auto *expr{std::get_if<ExtentExpr>(&value.u)};
      if (expr && expr->Rank() == 0) {
        result.emplace_back(std::move(*expr));
      } else {
        return std::nullopt;
      }
    }
    return result;
  } else {
    return std::nullopt;
  }
}

Shape GetShapeHelper::CreateShape(int rank, NamedEntity &base) {
  Shape shape;
  for (int dimension{0}; dimension < rank; ++dimension) {
    shape.emplace_back(GetExtent(base, dimension));
  }
  return shape;
}

std::optional<ExtentExpr> AsExtentArrayExpr(const Shape &shape) {
  ArrayConstructorValues<ExtentType> values;
  for (const auto &dim : shape) {
    if (dim) {
      values.Push(common::Clone(*dim));
    } else {
      return std::nullopt;
    }
  }
  return ExtentExpr{ArrayConstructor<ExtentType>{std::move(values)}};
}

std::optional<Constant<ExtentType>> AsConstantShape(
    FoldingContext &context, const Shape &shape) {
  if (auto shapeArray{AsExtentArrayExpr(shape)}) {
    auto folded{Fold(context, std::move(*shapeArray))};
    if (auto *p{UnwrapConstantValue<ExtentType>(folded)}) {
      return std::move(*p);
    }
  }
  return std::nullopt;
}

Constant<SubscriptInteger> AsConstantShape(const ConstantSubscripts &shape) {
  using IntType = Scalar<SubscriptInteger>;
  std::vector<IntType> result;
  for (auto dim : shape) {
    result.emplace_back(dim);
  }
  return {std::move(result), ConstantSubscripts{GetRank(shape)}};
}

ConstantSubscripts AsConstantExtents(const Constant<ExtentType> &shape) {
  ConstantSubscripts result;
  for (const auto &extent : shape.values()) {
    result.push_back(extent.ToInt64());
  }
  return result;
}

std::optional<ConstantSubscripts> AsConstantExtents(
    FoldingContext &context, const Shape &shape) {
  if (auto shapeConstant{AsConstantShape(context, shape)}) {
    return AsConstantExtents(*shapeConstant);
  } else {
    return std::nullopt;
  }
}

Shape AsShape(const ConstantSubscripts &shape) {
  Shape result;
  for (const auto &extent : shape) {
    result.emplace_back(ExtentExpr{extent});
  }
  return result;
}

std::optional<Shape> AsShape(const std::optional<ConstantSubscripts> &shape) {
  if (shape) {
    return AsShape(*shape);
  } else {
    return std::nullopt;
  }
}

Shape Fold(FoldingContext &context, Shape &&shape) {
  for (auto &dim : shape) {
    dim = Fold(context, std::move(dim));
  }
  return std::move(shape);
}

std::optional<Shape> Fold(
    FoldingContext &context, std::optional<Shape> &&shape) {
  if (shape) {
    return Fold(context, std::move(*shape));
  } else {
    return std::nullopt;
  }
}

static ExtentExpr ComputeTripCount(
    ExtentExpr &&lower, ExtentExpr &&upper, ExtentExpr &&stride) {
  ExtentExpr strideCopy{common::Clone(stride)};
  ExtentExpr span{
      (std::move(upper) - std::move(lower) + std::move(strideCopy)) /
      std::move(stride)};
  return ExtentExpr{
      Extremum<ExtentType>{Ordering::Greater, std::move(span), ExtentExpr{0}}};
}

ExtentExpr CountTrips(
    ExtentExpr &&lower, ExtentExpr &&upper, ExtentExpr &&stride) {
  return ComputeTripCount(
      std::move(lower), std::move(upper), std::move(stride));
}

ExtentExpr CountTrips(const ExtentExpr &lower, const ExtentExpr &upper,
    const ExtentExpr &stride) {
  return ComputeTripCount(
      common::Clone(lower), common::Clone(upper), common::Clone(stride));
}

MaybeExtentExpr CountTrips(MaybeExtentExpr &&lower, MaybeExtentExpr &&upper,
    MaybeExtentExpr &&stride) {
  std::function<ExtentExpr(ExtentExpr &&, ExtentExpr &&, ExtentExpr &&)> bound{
      std::bind(ComputeTripCount, _1, _2, _3)};
  return common::MapOptional(
      std::move(bound), std::move(lower), std::move(upper), std::move(stride));
}

MaybeExtentExpr GetSize(Shape &&shape) {
  ExtentExpr extent{1};
  for (auto &&dim : std::move(shape)) {
    if (dim) {
      extent = std::move(extent) * std::move(*dim);
    } else {
      return std::nullopt;
    }
  }
  return extent;
}

ConstantSubscript GetSize(const ConstantSubscripts &shape) {
  ConstantSubscript size{1};
  for (auto dim : shape) {
    CHECK(dim >= 0);
    size *= dim;
  }
  return size;
}

bool ContainsAnyImpliedDoIndex(const ExtentExpr &expr) {
  struct MyVisitor : public AnyTraverse<MyVisitor> {
    using Base = AnyTraverse<MyVisitor>;
    MyVisitor() : Base{*this} {}
    using Base::operator();
    bool operator()(const ImpliedDoIndex &) { return true; }
  };
  return MyVisitor{}(expr);
}

// Determines lower bound on a dimension.  This can be other than 1 only
// for a reference to a whole array object or component. (See LBOUND, 16.9.109).
// ASSOCIATE construct entities may require traversal of their referents.
template <typename RESULT, bool LBOUND_SEMANTICS>
class GetLowerBoundHelper
    : public Traverse<GetLowerBoundHelper<RESULT, LBOUND_SEMANTICS>, RESULT> {
public:
  using Result = RESULT;
  using Base = Traverse<GetLowerBoundHelper, RESULT>;
  using Base::operator();
  explicit GetLowerBoundHelper(int d, FoldingContext *context)
      : Base{*this}, dimension_{d}, context_{context} {}
  static Result Default() { return Result{1}; }
  static Result Combine(Result &&, Result &&) {
    // Operator results and array references always have lower bounds == 1
    return Result{1};
  }

  Result GetLowerBound(const Symbol &symbol0, NamedEntity &&base) const {
    const Symbol &symbol{symbol0.GetUltimate()};
    if (const auto *details{
            symbol.detailsIf<semantics::ObjectEntityDetails>()}) {
      int rank{details->shape().Rank()};
      if (dimension_ < rank) {
        const semantics::ShapeSpec &shapeSpec{details->shape()[dimension_]};
        if (shapeSpec.lbound().isExplicit()) {
          if (const auto &lbound{shapeSpec.lbound().GetExplicit()}) {
            if constexpr (LBOUND_SEMANTICS) {
              bool ok{false};
              auto lbValue{ToInt64(*lbound)};
              if (dimension_ == rank - 1 && details->IsAssumedSize()) {
                // last dimension of assumed-size dummy array: don't worry
                // about handling an empty dimension
                ok = IsScopeInvariantExpr(*lbound);
              } else if (lbValue.value_or(0) == 1) {
                // Lower bound is 1, regardless of extent
                ok = true;
              } else if (const auto &ubound{shapeSpec.ubound().GetExplicit()}) {
                // If we can't prove that the dimension is nonempty,
                // we must be conservative.
                // TODO: simple symbolic math in expression rewriting to
                // cope with cases like A(J:J)
                if (context_) {
                  auto extent{ToInt64(Fold(*context_,
                      ExtentExpr{*ubound} - ExtentExpr{*lbound} +
                          ExtentExpr{1}))};
                  if (extent) {
                    if (extent <= 0) {
                      return Result{1};
                    }
                    ok = true;
                  } else {
                    ok = false;
                  }
                } else {
                  auto ubValue{ToInt64(*ubound)};
                  if (lbValue && ubValue) {
                    if (*lbValue > *ubValue) {
                      return Result{1};
                    }
                    ok = true;
                  } else {
                    ok = false;
                  }
                }
              }
              return ok ? *lbound : Result{};
            } else {
              return *lbound;
            }
          } else {
            return Result{1};
          }
        }
        if (IsDescriptor(symbol)) {
          return ExtentExpr{DescriptorInquiry{std::move(base),
              DescriptorInquiry::Field::LowerBound, dimension_}};
        }
      }
    } else if (const auto *assoc{
                   symbol.detailsIf<semantics::AssocEntityDetails>()}) {
      if (assoc->rank()) { // SELECT RANK case
        const Symbol &resolved{ResolveAssociations(symbol)};
        if (IsDescriptor(resolved) && dimension_ < *assoc->rank()) {
          return ExtentExpr{DescriptorInquiry{std::move(base),
              DescriptorInquiry::Field::LowerBound, dimension_}};
        }
      } else {
        auto exprLowerBound{((*this)(assoc->expr()))};
        if (IsActuallyConstant(exprLowerBound)) {
          return std::move(exprLowerBound);
        } else {
          // If the lower bound of the associated entity is not resolved to
          // constant expression at the time of the association, it is unsafe
          // to re-evaluate it later in the associate construct. Statements
          // in-between may have modified its operands value.
          return ExtentExpr{DescriptorInquiry{std::move(base),
              DescriptorInquiry::Field::LowerBound, dimension_}};
        }
      }
    }
    if constexpr (LBOUND_SEMANTICS) {
      return Result{};
    } else {
      return Result{1};
    }
  }

  Result operator()(const Symbol &symbol0) const {
    return GetLowerBound(symbol0, NamedEntity{symbol0});
  }

  Result operator()(const Component &component) const {
    if (component.base().Rank() == 0) {
      return GetLowerBound(
          component.GetLastSymbol(), NamedEntity{common::Clone(component)});
    }
    return Result{1};
  }

private:
  int dimension_;
  FoldingContext *context_{nullptr};
};

ExtentExpr GetRawLowerBound(const NamedEntity &base, int dimension) {
  return GetLowerBoundHelper<ExtentExpr, false>{dimension, nullptr}(base);
}

ExtentExpr GetRawLowerBound(
    FoldingContext &context, const NamedEntity &base, int dimension) {
  return Fold(context,
      GetLowerBoundHelper<ExtentExpr, false>{dimension, &context}(base));
}

MaybeExtentExpr GetLBOUND(const NamedEntity &base, int dimension) {
  return GetLowerBoundHelper<MaybeExtentExpr, true>{dimension, nullptr}(base);
}

MaybeExtentExpr GetLBOUND(
    FoldingContext &context, const NamedEntity &base, int dimension) {
  return Fold(context,
      GetLowerBoundHelper<MaybeExtentExpr, true>{dimension, &context}(base));
}

Shape GetRawLowerBounds(const NamedEntity &base) {
  Shape result;
  int rank{base.Rank()};
  for (int dim{0}; dim < rank; ++dim) {
    result.emplace_back(GetRawLowerBound(base, dim));
  }
  return result;
}

Shape GetRawLowerBounds(FoldingContext &context, const NamedEntity &base) {
  Shape result;
  int rank{base.Rank()};
  for (int dim{0}; dim < rank; ++dim) {
    result.emplace_back(GetRawLowerBound(context, base, dim));
  }
  return result;
}

Shape GetLBOUNDs(const NamedEntity &base) {
  Shape result;
  int rank{base.Rank()};
  for (int dim{0}; dim < rank; ++dim) {
    result.emplace_back(GetLBOUND(base, dim));
  }
  return result;
}

Shape GetLBOUNDs(FoldingContext &context, const NamedEntity &base) {
  Shape result;
  int rank{base.Rank()};
  for (int dim{0}; dim < rank; ++dim) {
    result.emplace_back(GetLBOUND(context, base, dim));
  }
  return result;
}

// If the upper and lower bounds are constant, return a constant expression for
// the extent.  In particular, if the upper bound is less than the lower bound,
// return zero.
static MaybeExtentExpr GetNonNegativeExtent(
    const semantics::ShapeSpec &shapeSpec) {
  const auto &ubound{shapeSpec.ubound().GetExplicit()};
  const auto &lbound{shapeSpec.lbound().GetExplicit()};
  std::optional<ConstantSubscript> uval{ToInt64(ubound)};
  std::optional<ConstantSubscript> lval{ToInt64(lbound)};
  if (uval && lval) {
    if (*uval < *lval) {
      return ExtentExpr{0};
    } else {
      return ExtentExpr{*uval - *lval + 1};
    }
  } else if (lbound && ubound && IsScopeInvariantExpr(*lbound) &&
      IsScopeInvariantExpr(*ubound)) {
    // Apply effective IDIM (MAX calculation with 0) so thet the
    // result is never negative
    if (lval.value_or(0) == 1) {
      return ExtentExpr{Extremum<SubscriptInteger>{
          Ordering::Greater, ExtentExpr{0}, common::Clone(*ubound)}};
    } else {
      return ExtentExpr{
          Extremum<SubscriptInteger>{Ordering::Greater, ExtentExpr{0},
              common::Clone(*ubound) - common::Clone(*lbound) + ExtentExpr{1}}};
    }
  } else {
    return std::nullopt;
  }
}

MaybeExtentExpr GetAssociatedExtent(const NamedEntity &base,
    const semantics::AssocEntityDetails &assoc, int dimension) {
  if (auto shape{GetShape(assoc.expr())}) {
    if (dimension < static_cast<int>(shape->size())) {
      auto &extent{shape->at(dimension)};
      if (extent && IsActuallyConstant(*extent)) {
        return std::move(extent);
      } else {
        // Otherwise, evaluating the associated expression extent expression
        // after the associate statement is unsafe given statements inside the
        // associate may have modified the associated expression operands
        // values.
        return ExtentExpr{DescriptorInquiry{
            NamedEntity{base}, DescriptorInquiry::Field::Extent, dimension}};
      }
    }
  }
  return std::nullopt;
}

MaybeExtentExpr GetExtent(const NamedEntity &base, int dimension) {
  CHECK(dimension >= 0);
  const Symbol &last{base.GetLastSymbol()};
  const Symbol &symbol{ResolveAssociations(last)};
  if (const auto *assoc{last.detailsIf<semantics::AssocEntityDetails>()}) {
    if (assoc->rank()) { // SELECT RANK case
      if (semantics::IsDescriptor(symbol) && dimension < *assoc->rank()) {
        return ExtentExpr{DescriptorInquiry{
            NamedEntity{base}, DescriptorInquiry::Field::Extent, dimension}};
      }
    } else {
      return GetAssociatedExtent(base, *assoc, dimension);
    }
  }
  if (const auto *details{symbol.detailsIf<semantics::ObjectEntityDetails>()}) {
    if (IsImpliedShape(symbol) && details->init()) {
      if (auto shape{GetShape(symbol)}) {
        if (dimension < static_cast<int>(shape->size())) {
          return std::move(shape->at(dimension));
        }
      }
    } else {
      int j{0};
      for (const auto &shapeSpec : details->shape()) {
        if (j++ == dimension) {
          if (auto extent{GetNonNegativeExtent(shapeSpec)}) {
            return extent;
          } else if (details->IsAssumedSize() && j == symbol.Rank()) {
            return std::nullopt;
          } else if (semantics::IsDescriptor(symbol)) {
            return ExtentExpr{DescriptorInquiry{NamedEntity{base},
                DescriptorInquiry::Field::Extent, dimension}};
          } else {
            break;
          }
        }
      }
    }
  }
  return std::nullopt;
}

MaybeExtentExpr GetExtent(
    FoldingContext &context, const NamedEntity &base, int dimension) {
  return Fold(context, GetExtent(base, dimension));
}

MaybeExtentExpr GetExtent(
    const Subscript &subscript, const NamedEntity &base, int dimension) {
  return common::visit(
      common::visitors{
          [&](const Triplet &triplet) -> MaybeExtentExpr {
            MaybeExtentExpr upper{triplet.upper()};
            if (!upper) {
              upper = GetUBOUND(base, dimension);
            }
            MaybeExtentExpr lower{triplet.lower()};
            if (!lower) {
              lower = GetLBOUND(base, dimension);
            }
            return CountTrips(std::move(lower), std::move(upper),
                MaybeExtentExpr{triplet.stride()});
          },
          [&](const IndirectSubscriptIntegerExpr &subs) -> MaybeExtentExpr {
            if (auto shape{GetShape(subs.value())}) {
              if (GetRank(*shape) > 0) {
                CHECK(GetRank(*shape) == 1); // vector-valued subscript
                return std::move(shape->at(0));
              }
            }
            return std::nullopt;
          },
      },
      subscript.u);
}

MaybeExtentExpr GetExtent(FoldingContext &context, const Subscript &subscript,
    const NamedEntity &base, int dimension) {
  return Fold(context, GetExtent(subscript, base, dimension));
}

MaybeExtentExpr ComputeUpperBound(
    ExtentExpr &&lower, MaybeExtentExpr &&extent) {
  if (extent) {
    if (ToInt64(lower).value_or(0) == 1) {
      return std::move(*extent);
    } else {
      return std::move(*extent) + std::move(lower) - ExtentExpr{1};
    }
  } else {
    return std::nullopt;
  }
}

MaybeExtentExpr ComputeUpperBound(
    FoldingContext &context, ExtentExpr &&lower, MaybeExtentExpr &&extent) {
  return Fold(context, ComputeUpperBound(std::move(lower), std::move(extent)));
}

MaybeExtentExpr GetRawUpperBound(const NamedEntity &base, int dimension) {
  const Symbol &symbol{ResolveAssociations(base.GetLastSymbol())};
  if (const auto *details{symbol.detailsIf<semantics::ObjectEntityDetails>()}) {
    int rank{details->shape().Rank()};
    if (dimension < rank) {
      const auto &bound{details->shape()[dimension].ubound().GetExplicit()};
      if (bound && IsScopeInvariantExpr(*bound)) {
        return *bound;
      } else if (details->IsAssumedSize() && dimension + 1 == symbol.Rank()) {
        return std::nullopt;
      } else {
        return ComputeUpperBound(
            GetRawLowerBound(base, dimension), GetExtent(base, dimension));
      }
    }
  } else if (const auto *assoc{
                 symbol.detailsIf<semantics::AssocEntityDetails>()}) {
    if (auto extent{GetAssociatedExtent(base, *assoc, dimension)}) {
      return ComputeUpperBound(
          GetRawLowerBound(base, dimension), std::move(extent));
    }
  }
  return std::nullopt;
}

MaybeExtentExpr GetRawUpperBound(
    FoldingContext &context, const NamedEntity &base, int dimension) {
  return Fold(context, GetRawUpperBound(base, dimension));
}

static MaybeExtentExpr GetExplicitUBOUND(
    FoldingContext *context, const semantics::ShapeSpec &shapeSpec) {
  const auto &ubound{shapeSpec.ubound().GetExplicit()};
  if (ubound && IsScopeInvariantExpr(*ubound)) {
    if (auto extent{GetNonNegativeExtent(shapeSpec)}) {
      if (auto cstExtent{ToInt64(
              context ? Fold(*context, std::move(*extent)) : *extent)}) {
        if (cstExtent > 0) {
          return *ubound;
        } else if (cstExtent == 0) {
          return ExtentExpr{0};
        }
      }
    }
  }
  return std::nullopt;
}

static MaybeExtentExpr GetUBOUND(
    FoldingContext *context, const NamedEntity &base, int dimension) {
  const Symbol &symbol{ResolveAssociations(base.GetLastSymbol())};
  if (const auto *details{symbol.detailsIf<semantics::ObjectEntityDetails>()}) {
    int rank{details->shape().Rank()};
    if (dimension < rank) {
      const semantics::ShapeSpec &shapeSpec{details->shape()[dimension]};
      if (auto ubound{GetExplicitUBOUND(context, shapeSpec)}) {
        return *ubound;
      } else if (details->IsAssumedSize() && dimension + 1 == symbol.Rank()) {
        return std::nullopt;
      } else if (auto lb{GetLBOUND(base, dimension)}) {
        return ComputeUpperBound(std::move(*lb), GetExtent(base, dimension));
      }
    }
  } else if (const auto *assoc{
                 symbol.detailsIf<semantics::AssocEntityDetails>()}) {
    if (auto extent{GetAssociatedExtent(base, *assoc, dimension)}) {
      if (auto lb{GetLBOUND(base, dimension)}) {
        return ComputeUpperBound(std::move(*lb), std::move(extent));
      }
    }
  }
  return std::nullopt;
}

MaybeExtentExpr GetUBOUND(const NamedEntity &base, int dimension) {
  return GetUBOUND(nullptr, base, dimension);
}

MaybeExtentExpr GetUBOUND(
    FoldingContext &context, const NamedEntity &base, int dimension) {
  return Fold(context, GetUBOUND(&context, base, dimension));
}

static Shape GetUBOUNDs(FoldingContext *context, const NamedEntity &base) {
  const Symbol &symbol{ResolveAssociations(base.GetLastSymbol())};
  if (const auto *details{symbol.detailsIf<semantics::ObjectEntityDetails>()}) {
    Shape result;
    int dim{0};
    for (const auto &shapeSpec : details->shape()) {
      if (auto ubound{GetExplicitUBOUND(context, shapeSpec)}) {
        result.emplace_back(*ubound);
      } else if (details->IsAssumedSize() && dim + 1 == base.Rank()) {
        result.emplace_back(std::nullopt); // UBOUND folding replaces with -1
      } else if (auto lb{GetLBOUND(base, dim)}) {
        result.emplace_back(
            ComputeUpperBound(std::move(*lb), GetExtent(base, dim)));
      } else {
        result.emplace_back(); // unknown
      }
      ++dim;
    }
    CHECK(GetRank(result) == symbol.Rank());
    return result;
  } else {
    return std::move(GetShape(symbol).value());
  }
}

Shape GetUBOUNDs(FoldingContext &context, const NamedEntity &base) {
  return Fold(context, GetUBOUNDs(&context, base));
}

Shape GetUBOUNDs(const NamedEntity &base) { return GetUBOUNDs(nullptr, base); }

auto GetShapeHelper::operator()(const Symbol &symbol) const -> Result {
  return common::visit(
      common::visitors{
          [&](const semantics::ObjectEntityDetails &object) {
            if (IsImpliedShape(symbol) && object.init()) {
              return (*this)(object.init());
            } else if (IsAssumedRank(symbol)) {
              return Result{};
            } else {
              int n{object.shape().Rank()};
              NamedEntity base{symbol};
              return Result{CreateShape(n, base)};
            }
          },
          [](const semantics::EntityDetails &) {
            return ScalarShape(); // no dimensions seen
          },
          [&](const semantics::ProcEntityDetails &proc) {
            if (const Symbol * interface{proc.interface().symbol()}) {
              return (*this)(*interface);
            } else {
              return ScalarShape();
            }
          },
          [&](const semantics::AssocEntityDetails &assoc) {
            NamedEntity base{symbol};
            if (assoc.rank()) { // SELECT RANK case
              int n{assoc.rank().value()};
              return Result{CreateShape(n, base)};
            } else {
              auto exprShape{((*this)(assoc.expr()))};
              if (exprShape) {
                int rank{static_cast<int>(exprShape->size())};
                for (int dimension{0}; dimension < rank; ++dimension) {
                  auto &extent{(*exprShape)[dimension]};
                  if (extent && !IsActuallyConstant(*extent)) {
                    extent = GetExtent(base, dimension);
                  }
                }
              }
              return exprShape;
            }
          },
          [&](const semantics::SubprogramDetails &subp) -> Result {
            if (subp.isFunction()) {
              auto resultShape{(*this)(subp.result())};
              if (resultShape && !useResultSymbolShape_) {
                // Ensure the shape is constant. Otherwise, it may be referring
                // to symbols that belong to the subroutine scope and are
                // meaningless on the caller side without the related call
                // expression.
                for (auto &extent : *resultShape) {
                  if (extent && !IsActuallyConstant(*extent)) {
                    extent.reset();
                  }
                }
              }
              return resultShape;
            } else {
              return Result{};
            }
          },
          [&](const semantics::ProcBindingDetails &binding) {
            return (*this)(binding.symbol());
          },
          [](const semantics::TypeParamDetails &) { return ScalarShape(); },
          [](const auto &) { return Result{}; },
      },
      symbol.GetUltimate().details());
}

auto GetShapeHelper::operator()(const Component &component) const -> Result {
  const Symbol &symbol{component.GetLastSymbol()};
  int rank{symbol.Rank()};
  if (rank == 0) {
    return (*this)(component.base());
  } else if (symbol.has<semantics::ObjectEntityDetails>()) {
    NamedEntity base{Component{component}};
    return CreateShape(rank, base);
  } else if (symbol.has<semantics::AssocEntityDetails>()) {
    NamedEntity base{Component{component}};
    return Result{CreateShape(rank, base)};
  } else {
    return (*this)(symbol);
  }
}

auto GetShapeHelper::operator()(const ArrayRef &arrayRef) const -> Result {
  Shape shape;
  int dimension{0};
  const NamedEntity &base{arrayRef.base()};
  for (const Subscript &ss : arrayRef.subscript()) {
    if (ss.Rank() > 0) {
      shape.emplace_back(GetExtent(ss, base, dimension));
    }
    ++dimension;
  }
  if (shape.empty()) {
    if (const Component * component{base.UnwrapComponent()}) {
      return (*this)(component->base());
    }
  }
  return shape;
}

auto GetShapeHelper::operator()(const CoarrayRef &coarrayRef) const -> Result {
  NamedEntity base{coarrayRef.GetBase()};
  if (coarrayRef.subscript().empty()) {
    return (*this)(base);
  } else {
    Shape shape;
    int dimension{0};
    for (const Subscript &ss : coarrayRef.subscript()) {
      if (ss.Rank() > 0) {
        shape.emplace_back(GetExtent(ss, base, dimension));
      }
      ++dimension;
    }
    return shape;
  }
}

auto GetShapeHelper::operator()(const Substring &substring) const -> Result {
  return (*this)(substring.parent());
}

auto GetShapeHelper::operator()(const ProcedureRef &call) const -> Result {
  if (call.Rank() == 0) {
    return ScalarShape();
  } else if (call.IsElemental()) {
    for (const auto &arg : call.arguments()) {
      if (arg && arg->Rank() > 0) {
        return (*this)(*arg);
      }
    }
    return ScalarShape();
  } else if (const Symbol * symbol{call.proc().GetSymbol()}) {
    return (*this)(*symbol);
  } else if (const auto *intrinsic{call.proc().GetSpecificIntrinsic()}) {
    if (intrinsic->name == "shape" || intrinsic->name == "lbound" ||
        intrinsic->name == "ubound") {
      // For LBOUND/UBOUND, these are the array-valued cases (no DIM=)
      if (!call.arguments().empty() && call.arguments().front()) {
        return Shape{
            MaybeExtentExpr{ExtentExpr{call.arguments().front()->Rank()}}};
      }
    } else if (intrinsic->name == "all" || intrinsic->name == "any" ||
        intrinsic->name == "count" || intrinsic->name == "iall" ||
        intrinsic->name == "iany" || intrinsic->name == "iparity" ||
        intrinsic->name == "maxval" || intrinsic->name == "minval" ||
        intrinsic->name == "norm2" || intrinsic->name == "parity" ||
        intrinsic->name == "product" || intrinsic->name == "sum") {
      // Reduction with DIM=
      if (call.arguments().size() >= 2) {
        auto arrayShape{
            (*this)(UnwrapExpr<Expr<SomeType>>(call.arguments().at(0)))};
        const auto *dimArg{UnwrapExpr<Expr<SomeType>>(call.arguments().at(1))};
        if (arrayShape && dimArg) {
          if (auto dim{ToInt64(*dimArg)}) {
            if (*dim >= 1 &&
                static_cast<std::size_t>(*dim) <= arrayShape->size()) {
              arrayShape->erase(arrayShape->begin() + (*dim - 1));
              return std::move(*arrayShape);
            }
          }
        }
      }
    } else if (intrinsic->name == "findloc" || intrinsic->name == "maxloc" ||
        intrinsic->name == "minloc") {
      std::size_t dimIndex{intrinsic->name == "findloc" ? 2u : 1u};
      if (call.arguments().size() > dimIndex) {
        if (auto arrayShape{
                (*this)(UnwrapExpr<Expr<SomeType>>(call.arguments().at(0)))}) {
          auto rank{static_cast<int>(arrayShape->size())};
          if (const auto *dimArg{
                  UnwrapExpr<Expr<SomeType>>(call.arguments()[dimIndex])}) {
            auto dim{ToInt64(*dimArg)};
            if (dim && *dim >= 1 && *dim <= rank) {
              arrayShape->erase(arrayShape->begin() + (*dim - 1));
              return std::move(*arrayShape);
            }
          } else {
            // xxxLOC(no DIM=) result is vector(1:RANK(ARRAY=))
            return Shape{ExtentExpr{rank}};
          }
        }
      }
    } else if (intrinsic->name == "cshift" || intrinsic->name == "eoshift") {
      if (!call.arguments().empty()) {
        return (*this)(call.arguments()[0]);
      }
    } else if (intrinsic->name == "matmul") {
      if (call.arguments().size() == 2) {
        if (auto ashape{(*this)(call.arguments()[0])}) {
          if (auto bshape{(*this)(call.arguments()[1])}) {
            if (ashape->size() == 1 && bshape->size() == 2) {
              bshape->erase(bshape->begin());
              return std::move(*bshape); // matmul(vector, matrix)
            } else if (ashape->size() == 2 && bshape->size() == 1) {
              ashape->pop_back();
              return std::move(*ashape); // matmul(matrix, vector)
            } else if (ashape->size() == 2 && bshape->size() == 2) {
              (*ashape)[1] = std::move((*bshape)[1]);
              return std::move(*ashape); // matmul(matrix, matrix)
            }
          }
        }
      }
    } else if (intrinsic->name == "pack") {
      if (call.arguments().size() >= 3 && call.arguments().at(2)) {
        // SHAPE(PACK(,,VECTOR=v)) -> SHAPE(v)
        return (*this)(call.arguments().at(2));
      } else if (call.arguments().size() >= 2 && context_) {
        if (auto maskShape{(*this)(call.arguments().at(1))}) {
          if (maskShape->size() == 0) {
            // Scalar MASK= -> [MERGE(SIZE(ARRAY=), 0, mask)]
            if (auto arrayShape{(*this)(call.arguments().at(0))}) {
              auto arraySize{GetSize(std::move(*arrayShape))};
              CHECK(arraySize);
              ActualArguments toMerge{
                  ActualArgument{AsGenericExpr(std::move(*arraySize))},
                  ActualArgument{AsGenericExpr(ExtentExpr{0})},
                  common::Clone(call.arguments().at(1))};
              auto specific{context_->intrinsics().Probe(
                  CallCharacteristics{"merge"}, toMerge, *context_)};
              CHECK(specific);
              return Shape{ExtentExpr{FunctionRef<ExtentType>{
                  ProcedureDesignator{std::move(specific->specificIntrinsic)},
                  std::move(specific->arguments)}}};
            }
          } else {
            // Non-scalar MASK= -> [COUNT(mask)]
            ActualArguments toCount{ActualArgument{common::Clone(
                DEREF(call.arguments().at(1).value().UnwrapExpr()))}};
            auto specific{context_->intrinsics().Probe(
                CallCharacteristics{"count"}, toCount, *context_)};
            CHECK(specific);
            return Shape{ExtentExpr{FunctionRef<ExtentType>{
                ProcedureDesignator{std::move(specific->specificIntrinsic)},
                std::move(specific->arguments)}}};
          }
        }
      }
    } else if (intrinsic->name == "reshape") {
      if (call.arguments().size() >= 2 && call.arguments().at(1)) {
        // SHAPE(RESHAPE(array,shape)) -> shape
        if (const auto *shapeExpr{
                call.arguments().at(1).value().UnwrapExpr()}) {
          auto shapeArg{std::get<Expr<SomeInteger>>(shapeExpr->u)};
          if (auto result{AsShapeResult(
                  ConvertToType<ExtentType>(std::move(shapeArg)))}) {
            return result;
          }
        }
      }
    } else if (intrinsic->name == "spread") {
      // SHAPE(SPREAD(ARRAY,DIM,NCOPIES)) = SHAPE(ARRAY) with NCOPIES inserted
      // at position DIM.
      if (call.arguments().size() == 3) {
        auto arrayShape{
            (*this)(UnwrapExpr<Expr<SomeType>>(call.arguments().at(0)))};
        const auto *dimArg{UnwrapExpr<Expr<SomeType>>(call.arguments().at(1))};
        const auto *nCopies{
            UnwrapExpr<Expr<SomeInteger>>(call.arguments().at(2))};
        if (arrayShape && dimArg && nCopies) {
          if (auto dim{ToInt64(*dimArg)}) {
            if (*dim >= 1 &&
                static_cast<std::size_t>(*dim) <= arrayShape->size() + 1) {
              arrayShape->emplace(arrayShape->begin() + *dim - 1,
                  ConvertToType<ExtentType>(common::Clone(*nCopies)));
              return std::move(*arrayShape);
            }
          }
        }
      }
    } else if (intrinsic->name == "transfer") {
      if (call.arguments().size() == 3 && call.arguments().at(2)) {
        // SIZE= is present; shape is vector [SIZE=]
        if (const auto *size{
                UnwrapExpr<Expr<SomeInteger>>(call.arguments().at(2))}) {
          return Shape{
              MaybeExtentExpr{ConvertToType<ExtentType>(common::Clone(*size))}};
        }
      } else if (context_) {
        if (auto moldTypeAndShape{characteristics::TypeAndShape::Characterize(
                call.arguments().at(1), *context_)}) {
          if (GetRank(moldTypeAndShape->shape()) == 0) {
            // SIZE= is absent and MOLD= is scalar: result is scalar
            return ScalarShape();
          } else {
            // SIZE= is absent and MOLD= is array: result is vector whose
            // length is determined by sizes of types.  See 16.9.193p4 case(ii).
            // Note that if sourceBytes is not known to be empty, we
            // can fold only when moldElementBytes is known to not be zero;
            // the most general case risks a division by zero otherwise.
            if (auto sourceTypeAndShape{
                    characteristics::TypeAndShape::Characterize(
                        call.arguments().at(0), *context_)}) {
              if (auto sourceBytes{
                      sourceTypeAndShape->MeasureSizeInBytes(*context_)}) {
                *sourceBytes = Fold(*context_, std::move(*sourceBytes));
                if (auto sourceBytesConst{ToInt64(*sourceBytes)}) {
                  if (*sourceBytesConst == 0) {
                    return Shape{ExtentExpr{0}};
                  }
                }
                if (auto moldElementBytes{
                        moldTypeAndShape->MeasureElementSizeInBytes(
                            *context_, true)}) {
                  *moldElementBytes =
                      Fold(*context_, std::move(*moldElementBytes));
                  auto moldElementBytesConst{ToInt64(*moldElementBytes)};
                  if (moldElementBytesConst && *moldElementBytesConst != 0) {
                    ExtentExpr extent{Fold(*context_,
                        (std::move(*sourceBytes) +
                            common::Clone(*moldElementBytes) - ExtentExpr{1}) /
                            common::Clone(*moldElementBytes))};
                    return Shape{MaybeExtentExpr{std::move(extent)}};
                  }
                }
              }
            }
          }
        }
      }
    } else if (intrinsic->name == "transpose") {
      if (call.arguments().size() >= 1) {
        if (auto shape{(*this)(call.arguments().at(0))}) {
          if (shape->size() == 2) {
            std::swap((*shape)[0], (*shape)[1]);
            return shape;
          }
        }
      }
    } else if (intrinsic->name == "unpack") {
      if (call.arguments().size() >= 2) {
        return (*this)(call.arguments()[1]); // MASK=
      }
    } else if (intrinsic->characteristics.value().attrs.test(characteristics::
                       Procedure::Attr::NullPointer)) { // NULL(MOLD=)
      return (*this)(call.arguments());
    } else {
      // TODO: shapes of other non-elemental intrinsic results
    }
  }
  // The rank is always known even if the extents are not.
  return Shape(static_cast<std::size_t>(call.Rank()), MaybeExtentExpr{});
}

// Check conformance of the passed shapes.
std::optional<bool> CheckConformance(parser::ContextualMessages &messages,
    const Shape &left, const Shape &right, CheckConformanceFlags::Flags flags,
    const char *leftIs, const char *rightIs) {
  int n{GetRank(left)};
  if (n == 0 && (flags & CheckConformanceFlags::LeftScalarExpandable)) {
    return true;
  }
  int rn{GetRank(right)};
  if (rn == 0 && (flags & CheckConformanceFlags::RightScalarExpandable)) {
    return true;
  }
  if (n != rn) {
    messages.Say("Rank of %1$s is %2$d, but %3$s has rank %4$d"_err_en_US,
        leftIs, n, rightIs, rn);
    return false;
  }
  for (int j{0}; j < n; ++j) {
    if (auto leftDim{ToInt64(left[j])}) {
      if (auto rightDim{ToInt64(right[j])}) {
        if (*leftDim != *rightDim) {
          messages.Say("Dimension %1$d of %2$s has extent %3$jd, "
                       "but %4$s has extent %5$jd"_err_en_US,
              j + 1, leftIs, *leftDim, rightIs, *rightDim);
          return false;
        }
      } else if (!(flags & CheckConformanceFlags::RightIsDeferredShape)) {
        return std::nullopt;
      }
    } else if (!(flags & CheckConformanceFlags::LeftIsDeferredShape)) {
      return std::nullopt;
    }
  }
  return true;
}

bool IncrementSubscripts(
    ConstantSubscripts &indices, const ConstantSubscripts &extents) {
  std::size_t rank(indices.size());
  CHECK(rank <= extents.size());
  for (std::size_t j{0}; j < rank; ++j) {
    if (extents[j] < 1) {
      return false;
    }
  }
  for (std::size_t j{0}; j < rank; ++j) {
    if (indices[j]++ < extents[j]) {
      return true;
    }
    indices[j] = 1;
  }
  return false;
}

} // namespace Fortran::evaluate
