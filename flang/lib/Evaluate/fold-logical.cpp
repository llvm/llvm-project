//===-- lib/Evaluate/fold-logical.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "fold-implementation.h"
#include "flang/Evaluate/check-expression.h"

namespace Fortran::evaluate {

template <int KIND>
Expr<Type<TypeCategory::Logical, KIND>> FoldIntrinsicFunction(
    FoldingContext &context,
    FunctionRef<Type<TypeCategory::Logical, KIND>> &&funcRef) {
  using T = Type<TypeCategory::Logical, KIND>;
  ActualArguments &args{funcRef.arguments()};
  auto *intrinsic{std::get_if<SpecificIntrinsic>(&funcRef.proc().u)};
  CHECK(intrinsic);
  std::string name{intrinsic->name};
  if (name == "all") {
    if (!args[1]) { // TODO: ALL(x,DIM=d)
      if (const auto *constant{UnwrapConstantValue<T>(args[0])}) {
        bool result{true};
        for (const auto &element : constant->values()) {
          if (!element.IsTrue()) {
            result = false;
            break;
          }
        }
        return Expr<T>{result};
      }
    }
  } else if (name == "any") {
    if (!args[1]) { // TODO: ANY(x,DIM=d)
      if (const auto *constant{UnwrapConstantValue<T>(args[0])}) {
        bool result{false};
        for (const auto &element : constant->values()) {
          if (element.IsTrue()) {
            result = true;
            break;
          }
        }
        return Expr<T>{result};
      }
    }
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
    using LargestInt = Type<TypeCategory::Integer, 16>;
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
  } else if (name == "isnan") {
    // A warning about an invalid argument is discarded from converting
    // the argument of isnan().
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
  // TODO: btest, cshift, dot_product, eoshift, is_iostat_end,
  // is_iostat_eor, lge, lgt, lle, llt, logical, matmul, out_of_range,
  // pack, parity, reduce, spread, transfer, transpose, unpack,
  // extends_type_of, same_type_as
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
