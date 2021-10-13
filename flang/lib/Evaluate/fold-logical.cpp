//===-- lib/Evaluate/fold-logical.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "fold-implementation.h"
#include "fold-reduction.h"
#include "flang/Evaluate/check-expression.h"

namespace Fortran::evaluate {

// for ALL & ANY
template <typename T>
static Expr<T> FoldAllAny(FoldingContext &context, FunctionRef<T> &&ref,
    Scalar<T> (Scalar<T>::*operation)(const Scalar<T> &) const,
    Scalar<T> identity) {
  static_assert(T::category == TypeCategory::Logical);
  using Element = Scalar<T>;
  std::optional<int> dim;
  if (std::optional<Constant<T>> array{
          ProcessReductionArgs<T>(context, ref.arguments(), dim, identity,
              /*ARRAY(MASK)=*/0, /*DIM=*/1)}) {
    auto accumulator{[&](Element &element, const ConstantSubscripts &at) {
      element = (element.*operation)(array->At(at));
    }};
    return Expr<T>{DoReduction<T>(*array, dim, identity, accumulator)};
  }
  return Expr<T>{std::move(ref)};
}

template <int KIND>
Expr<Type<TypeCategory::Logical, KIND>> FoldIntrinsicFunction(
    FoldingContext &context,
    FunctionRef<Type<TypeCategory::Logical, KIND>> &&funcRef) {
  using T = Type<TypeCategory::Logical, KIND>;
  ActualArguments &args{funcRef.arguments()};
  auto *intrinsic{std::get_if<SpecificIntrinsic>(&funcRef.proc().u)};
  CHECK(intrinsic);
  std::string name{intrinsic->name};
  using SameInt = Type<TypeCategory::Integer, KIND>;
  if (name == "all") {
    return FoldAllAny(
        context, std::move(funcRef), &Scalar<T>::AND, Scalar<T>{true});
  } else if (name == "any") {
    return FoldAllAny(
        context, std::move(funcRef), &Scalar<T>::OR, Scalar<T>{false});
  } else if (name == "associated") {
    bool gotConstant{true};
    const Expr<SomeType> *firstArgExpr{args[0]->UnwrapExpr()};
    if (!firstArgExpr || !IsNullPointer(*firstArgExpr)) {
      gotConstant = false;
    } else if (args[1]) { // There's a second argument
      const Expr<SomeType> *secondArgExpr{args[1]->UnwrapExpr()};
      if (!secondArgExpr || !IsNullPointer(*secondArgExpr)) {
        gotConstant = false;
      }
    }
    return gotConstant ? Expr<T>{false} : Expr<T>{std::move(funcRef)};
  } else if (name == "bge" || name == "bgt" || name == "ble" || name == "blt") {
    static_assert(std::is_same_v<Scalar<LargestInt>, BOZLiteralConstant>);
    // Arguments do not have to be of the same integer type. Convert all
    // arguments to the biggest integer type before comparing them to
    // simplify.
    for (int i{0}; i <= 1; ++i) {
      if (auto *x{UnwrapExpr<Expr<SomeInteger>>(args[i])}) {
        *args[i] = AsGenericExpr(
            Fold(context, ConvertToType<LargestInt>(std::move(*x))));
      } else if (auto *x{UnwrapExpr<BOZLiteralConstant>(args[i])}) {
        *args[i] = AsGenericExpr(Constant<LargestInt>{std::move(*x)});
      }
    }
    auto fptr{&Scalar<LargestInt>::BGE};
    if (name == "bge") { // done in fptr declaration
    } else if (name == "bgt") {
      fptr = &Scalar<LargestInt>::BGT;
    } else if (name == "ble") {
      fptr = &Scalar<LargestInt>::BLE;
    } else if (name == "blt") {
      fptr = &Scalar<LargestInt>::BLT;
    } else {
      common::die("missing case to fold intrinsic function %s", name.c_str());
    }
    return FoldElementalIntrinsic<T, LargestInt, LargestInt>(context,
        std::move(funcRef),
        ScalarFunc<T, LargestInt, LargestInt>(
            [&fptr](const Scalar<LargestInt> &i, const Scalar<LargestInt> &j) {
              return Scalar<T>{std::invoke(fptr, i, j)};
            }));
  } else if (name == "btest") {
    if (const auto *ix{UnwrapExpr<Expr<SomeInteger>>(args[0])}) {
      return std::visit(
          [&](const auto &x) {
            using IT = ResultType<decltype(x)>;
            return FoldElementalIntrinsic<T, IT, SameInt>(context,
                std::move(funcRef),
                ScalarFunc<T, IT, SameInt>(
                    [&](const Scalar<IT> &x, const Scalar<SameInt> &pos) {
                      auto posVal{pos.ToInt64()};
                      if (posVal < 0 || posVal >= x.bits) {
                        context.messages().Say(
                            "POS=%jd out of range for BTEST"_err_en_US,
                            static_cast<std::intmax_t>(posVal));
                      }
                      return Scalar<T>{x.BTEST(posVal)};
                    }));
          },
          ix->u);
    }
  } else if (name == "isnan" || name == "__builtin_ieee_is_nan") {
    // A warning about an invalid argument is discarded from converting
    // the argument of isnan() / IEEE_IS_NAN().
    auto restorer{context.messages().DiscardMessages()};
    using DefaultReal = Type<TypeCategory::Real, 4>;
    return FoldElementalIntrinsic<T, DefaultReal>(context, std::move(funcRef),
        ScalarFunc<T, DefaultReal>([](const Scalar<DefaultReal> &x) {
          return Scalar<T>{x.IsNotANumber()};
        }));
  } else if (name == "is_contiguous") {
    if (args.at(0)) {
      if (auto *expr{args[0]->UnwrapExpr()}) {
        if (IsSimplyContiguous(*expr, context)) {
          return Expr<T>{true};
        }
      }
    }
  } else if (name == "lge" || name == "lgt" || name == "lle" || name == "llt") {
    // Rewrite LGE/LGT/LLE/LLT into ASCII character relations
    auto *cx0{UnwrapExpr<Expr<SomeCharacter>>(args[0])};
    auto *cx1{UnwrapExpr<Expr<SomeCharacter>>(args[1])};
    if (cx0 && cx1) {
      return Fold(context,
          ConvertToType<T>(
              PackageRelation(name == "lge" ? RelationalOperator::GE
                      : name == "lgt"       ? RelationalOperator::GT
                      : name == "lle"       ? RelationalOperator::LE
                                            : RelationalOperator::LT,
                  ConvertToType<Ascii>(std::move(*cx0)),
                  ConvertToType<Ascii>(std::move(*cx1)))));
    }
  } else if (name == "logical") {
    if (auto *expr{UnwrapExpr<Expr<SomeLogical>>(args[0])}) {
      return Fold(context, ConvertToType<T>(std::move(*expr)));
    }
  } else if (name == "merge") {
    return FoldMerge<T>(context, std::move(funcRef));
  } else if (name == "__builtin_ieee_support_datatype" ||
      name == "__builtin_ieee_support_denormal" ||
      name == "__builtin_ieee_support_divide" ||
      name == "__builtin_ieee_support_divide" ||
      name == "__builtin_ieee_support_inf" ||
      name == "__builtin_ieee_support_io" ||
      name == "__builtin_ieee_support_nan" ||
      name == "__builtin_ieee_support_sqrt" ||
      name == "__builtin_ieee_support_standard" ||
      name == "__builtin_ieee_support_subnormal" ||
      name == "__builtin_ieee_support_underflow_control") {
    return Expr<T>{true};
  }
  // TODO: dot_product, is_iostat_end,
  // is_iostat_eor, logical, matmul, out_of_range,
  // parity, transfer
  return Expr<T>{std::move(funcRef)};
}

template <typename T>
Expr<LogicalResult> FoldOperation(
    FoldingContext &context, Relational<T> &&relation) {
  if (auto array{ApplyElementwise(context, relation,
          std::function<Expr<LogicalResult>(Expr<T> &&, Expr<T> &&)>{
              [=](Expr<T> &&x, Expr<T> &&y) {
                return Expr<LogicalResult>{Relational<SomeType>{
                    Relational<T>{relation.opr, std::move(x), std::move(y)}}};
              }})}) {
    return *array;
  }
  if (auto folded{OperandsAreConstants(relation)}) {
    bool result{};
    if constexpr (T::category == TypeCategory::Integer) {
      result =
          Satisfies(relation.opr, folded->first.CompareSigned(folded->second));
    } else if constexpr (T::category == TypeCategory::Real) {
      result = Satisfies(relation.opr, folded->first.Compare(folded->second));
    } else if constexpr (T::category == TypeCategory::Complex) {
      result = (relation.opr == RelationalOperator::EQ) ==
          folded->first.Equals(folded->second);
    } else if constexpr (T::category == TypeCategory::Character) {
      result = Satisfies(relation.opr, Compare(folded->first, folded->second));
    } else {
      static_assert(T::category != TypeCategory::Logical);
    }
    return Expr<LogicalResult>{Constant<LogicalResult>{result}};
  }
  return Expr<LogicalResult>{Relational<SomeType>{std::move(relation)}};
}

Expr<LogicalResult> FoldOperation(
    FoldingContext &context, Relational<SomeType> &&relation) {
  return std::visit(
      [&](auto &&x) {
        return Expr<LogicalResult>{FoldOperation(context, std::move(x))};
      },
      std::move(relation.u));
}

template <int KIND>
Expr<Type<TypeCategory::Logical, KIND>> FoldOperation(
    FoldingContext &context, Not<KIND> &&x) {
  if (auto array{ApplyElementwise(context, x)}) {
    return *array;
  }
  using Ty = Type<TypeCategory::Logical, KIND>;
  auto &operand{x.left()};
  if (auto value{GetScalarConstantValue<Ty>(operand)}) {
    return Expr<Ty>{Constant<Ty>{!value->IsTrue()}};
  }
  return Expr<Ty>{x};
}

template <int KIND>
Expr<Type<TypeCategory::Logical, KIND>> FoldOperation(
    FoldingContext &context, LogicalOperation<KIND> &&operation) {
  using LOGICAL = Type<TypeCategory::Logical, KIND>;
  if (auto array{ApplyElementwise(context, operation,
          std::function<Expr<LOGICAL>(Expr<LOGICAL> &&, Expr<LOGICAL> &&)>{
              [=](Expr<LOGICAL> &&x, Expr<LOGICAL> &&y) {
                return Expr<LOGICAL>{LogicalOperation<KIND>{
                    operation.logicalOperator, std::move(x), std::move(y)}};
              }})}) {
    return *array;
  }
  if (auto folded{OperandsAreConstants(operation)}) {
    bool xt{folded->first.IsTrue()}, yt{folded->second.IsTrue()}, result{};
    switch (operation.logicalOperator) {
    case LogicalOperator::And:
      result = xt && yt;
      break;
    case LogicalOperator::Or:
      result = xt || yt;
      break;
    case LogicalOperator::Eqv:
      result = xt == yt;
      break;
    case LogicalOperator::Neqv:
      result = xt != yt;
      break;
    case LogicalOperator::Not:
      DIE("not a binary operator");
    }
    return Expr<LOGICAL>{Constant<LOGICAL>{result}};
  }
  return Expr<LOGICAL>{std::move(operation)};
}

FOR_EACH_LOGICAL_KIND(template class ExpressionBase, )
template class ExpressionBase<SomeLogical>;
} // namespace Fortran::evaluate
