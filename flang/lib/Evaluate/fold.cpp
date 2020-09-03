//===-- lib/Evaluate/fold.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Evaluate/fold.h"
#include "fold-implementation.h"

namespace Fortran::evaluate {

std::optional<Constant<SubscriptInteger>> GetConstantSubscript(
    FoldingContext &context, Subscript &ss, const NamedEntity &base, int dim) {
  ss = FoldOperation(context, std::move(ss));
  return std::visit(
      common::visitors{
          [](IndirectSubscriptIntegerExpr &expr)
              -> std::optional<Constant<SubscriptInteger>> {
            if (const auto *constant{
                    UnwrapConstantValue<SubscriptInteger>(expr.value())}) {
              return *constant;
            } else {
              return std::nullopt;
            }
          },
          [&](Triplet &triplet) -> std::optional<Constant<SubscriptInteger>> {
            auto lower{triplet.lower()}, upper{triplet.upper()};
            std::optional<ConstantSubscript> stride{ToInt64(triplet.stride())};
            if (!lower) {
              lower = GetLowerBound(context, base, dim);
            }
            if (!upper) {
              upper =
                  ComputeUpperBound(context, GetLowerBound(context, base, dim),
                      GetExtent(context, base, dim));
            }
            auto lbi{ToInt64(lower)}, ubi{ToInt64(upper)};
            if (lbi && ubi && stride && *stride != 0) {
              std::vector<SubscriptInteger::Scalar> values;
              while ((*stride > 0 && *lbi <= *ubi) ||
                  (*stride < 0 && *lbi >= *ubi)) {
                values.emplace_back(*lbi);
                *lbi += *stride;
              }
              return Constant<SubscriptInteger>{std::move(values),
                  ConstantSubscripts{
                      static_cast<ConstantSubscript>(values.size())}};
            } else {
              return std::nullopt;
            }
          },
      },
      ss.u);
}

// TODO: Put this in a more central location if it would be useful elsewhere
class ScalarConstantExpander {
public:
  explicit ScalarConstantExpander(ConstantSubscripts &extents)
      : extents_{extents} {}

  template <typename A> A Expand(A &&x) const {
    return std::move(x); // default case
  }
  template <typename T> Constant<T> Expand(Constant<T> &&x) {
    return x.Reshape(std::move(extents_));
  }
  template <typename T> Expr<T> Expand(Expr<T> &&x) {
    return std::visit([&](auto &&x) { return Expr<T>{Expand(std::move(x))}; },
        std::move(x.u));
  }

private:
  ConstantSubscripts &extents_;
};

Expr<SomeDerived> FoldOperation(
    FoldingContext &context, StructureConstructor &&structure) {
  StructureConstructor ctor{structure.derivedTypeSpec()};
  bool constantExtents{true};
  for (auto &&[symbol, value] : std::move(structure)) {
    auto expr{Fold(context, std::move(value.value()))};
    if (!IsProcedurePointer(symbol)) {
      if (auto valueShape{GetConstantExtents(context, expr)}) {
        if (!IsPointer(symbol)) {
          if (auto componentShape{GetConstantExtents(context, symbol)}) {
            if (GetRank(*componentShape) > 0 && GetRank(*valueShape) == 0) {
              expr = ScalarConstantExpander{*componentShape}.Expand(
                  std::move(expr));
              constantExtents = constantExtents && expr.Rank() > 0;
            } else {
              constantExtents =
                  constantExtents && *valueShape == *componentShape;
            }
          } else {
            constantExtents = false;
          }
        }
      } else {
        constantExtents = false;
      }
    }
    ctor.Add(symbol, Fold(context, std::move(expr)));
  }
  if (constantExtents && IsConstantExpr(ctor)) {
    return Expr<SomeDerived>{Constant<SomeDerived>{std::move(ctor)}};
  } else {
    return Expr<SomeDerived>{std::move(ctor)};
  }
}

Component FoldOperation(FoldingContext &context, Component &&component) {
  return {FoldOperation(context, std::move(component.base())),
      component.GetLastSymbol()};
}

NamedEntity FoldOperation(FoldingContext &context, NamedEntity &&x) {
  if (Component * c{x.UnwrapComponent()}) {
    return NamedEntity{FoldOperation(context, std::move(*c))};
  } else {
    return std::move(x);
  }
}

Triplet FoldOperation(FoldingContext &context, Triplet &&triplet) {
  MaybeExtentExpr lower{triplet.lower()};
  MaybeExtentExpr upper{triplet.upper()};
  return {Fold(context, std::move(lower)), Fold(context, std::move(upper)),
      Fold(context, triplet.stride())};
}

Subscript FoldOperation(FoldingContext &context, Subscript &&subscript) {
  return std::visit(common::visitors{
                        [&](IndirectSubscriptIntegerExpr &&expr) {
                          expr.value() = Fold(context, std::move(expr.value()));
                          return Subscript(std::move(expr));
                        },
                        [&](Triplet &&triplet) {
                          return Subscript(
                              FoldOperation(context, std::move(triplet)));
                        },
                    },
      std::move(subscript.u));
}

ArrayRef FoldOperation(FoldingContext &context, ArrayRef &&arrayRef) {
  NamedEntity base{FoldOperation(context, std::move(arrayRef.base()))};
  for (Subscript &subscript : arrayRef.subscript()) {
    subscript = FoldOperation(context, std::move(subscript));
  }
  return ArrayRef{std::move(base), std::move(arrayRef.subscript())};
}

CoarrayRef FoldOperation(FoldingContext &context, CoarrayRef &&coarrayRef) {
  std::vector<Subscript> subscript;
  for (Subscript x : coarrayRef.subscript()) {
    subscript.emplace_back(FoldOperation(context, std::move(x)));
  }
  std::vector<Expr<SubscriptInteger>> cosubscript;
  for (Expr<SubscriptInteger> x : coarrayRef.cosubscript()) {
    cosubscript.emplace_back(Fold(context, std::move(x)));
  }
  CoarrayRef folded{std::move(coarrayRef.base()), std::move(subscript),
      std::move(cosubscript)};
  if (std::optional<Expr<SomeInteger>> stat{coarrayRef.stat()}) {
    folded.set_stat(Fold(context, std::move(*stat)));
  }
  if (std::optional<Expr<SomeInteger>> team{coarrayRef.team()}) {
    folded.set_team(
        Fold(context, std::move(*team)), coarrayRef.teamIsTeamNumber());
  }
  return folded;
}

DataRef FoldOperation(FoldingContext &context, DataRef &&dataRef) {
  return std::visit(common::visitors{
                        [&](SymbolRef symbol) { return DataRef{*symbol}; },
                        [&](auto &&x) {
                          return DataRef{FoldOperation(context, std::move(x))};
                        },
                    },
      std::move(dataRef.u));
}

Substring FoldOperation(FoldingContext &context, Substring &&substring) {
  auto lower{Fold(context, substring.lower())};
  auto upper{Fold(context, substring.upper())};
  if (const DataRef * dataRef{substring.GetParentIf<DataRef>()}) {
    return Substring{FoldOperation(context, DataRef{*dataRef}),
        std::move(lower), std::move(upper)};
  } else {
    auto p{*substring.GetParentIf<StaticDataObject::Pointer>()};
    return Substring{std::move(p), std::move(lower), std::move(upper)};
  }
}

ComplexPart FoldOperation(FoldingContext &context, ComplexPart &&complexPart) {
  DataRef complex{complexPart.complex()};
  return ComplexPart{
      FoldOperation(context, std::move(complex)), complexPart.part()};
}

std::optional<std::int64_t> GetInt64Arg(
    const std::optional<ActualArgument> &arg) {
  if (const auto *intExpr{UnwrapExpr<Expr<SomeInteger>>(arg)}) {
    return ToInt64(*intExpr);
  } else {
    return std::nullopt;
  }
}

std::optional<std::int64_t> GetInt64ArgOr(
    const std::optional<ActualArgument> &arg, std::int64_t defaultValue) {
  if (!arg) {
    return defaultValue;
  } else if (const auto *intExpr{UnwrapExpr<Expr<SomeInteger>>(arg)}) {
    return ToInt64(*intExpr);
  } else {
    return std::nullopt;
  }
}

Expr<ImpliedDoIndex::Result> FoldOperation(
    FoldingContext &context, ImpliedDoIndex &&iDo) {
  if (std::optional<ConstantSubscript> value{context.GetImpliedDo(iDo.name)}) {
    return Expr<ImpliedDoIndex::Result>{*value};
  } else {
    return Expr<ImpliedDoIndex::Result>{std::move(iDo)};
  }
}

template class ExpressionBase<SomeDerived>;
template class ExpressionBase<SomeType>;

} // namespace Fortran::evaluate
