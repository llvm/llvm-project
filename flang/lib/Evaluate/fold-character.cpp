//===-- lib/Evaluate/fold-character.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "fold-implementation.h"
#include "fold-reduction.h"

namespace Fortran::evaluate {

static std::optional<ConstantSubscript> GetConstantLength(
    FoldingContext &context, Expr<SomeType> &&expr) {
  expr = Fold(context, std::move(expr));
  if (auto *chExpr{UnwrapExpr<Expr<SomeCharacter>>(expr)}) {
    if (auto len{chExpr->LEN()}) {
      return ToInt64(*len);
    }
  }
  return std::nullopt;
}

template <typename T>
static std::optional<ConstantSubscript> GetConstantLength(
    FoldingContext &context, FunctionRef<T> &funcRef, int zeroBasedArg) {
  if (auto *expr{funcRef.UnwrapArgExpr(zeroBasedArg)}) {
    return GetConstantLength(context, std::move(*expr));
  } else {
    return std::nullopt;
  }
}

template <typename T>
static std::optional<Scalar<T>> Identity(
    Scalar<T> str, std::optional<ConstantSubscript> len) {
  if (len) {
    return CharacterUtils<T::kind>::REPEAT(
        str, std::max<ConstantSubscript>(*len, 0));
  } else {
    return std::nullopt;
  }
}

template <int KIND>
Expr<Type<TypeCategory::Character, KIND>> FoldIntrinsicFunction(
    FoldingContext &context,
    FunctionRef<Type<TypeCategory::Character, KIND>> &&funcRef) {
  using T = Type<TypeCategory::Character, KIND>;
  using StringType = Scalar<T>; // std::string or larger
  using SingleCharType = typename StringType::value_type; // char &c.
  auto *intrinsic{std::get_if<SpecificIntrinsic>(&funcRef.proc().u)};
  CHECK(intrinsic);
  std::string name{intrinsic->name};
  if (name == "achar" || name == "char") {
    using IntT = SubscriptInteger;
    return FoldElementalIntrinsic<T, IntT>(context, std::move(funcRef),
        ScalarFunc<T, IntT>([&](const Scalar<IntT> &i) {
          if (i.IsNegative() || i.BGE(Scalar<IntT>{0}.IBSET(8 * KIND))) {
            if (context.languageFeatures().ShouldWarn(
                    common::UsageWarning::FoldingValueChecks)) {
              context.messages().Say(common::UsageWarning::FoldingValueChecks,
                  "%s(I=%jd) is out of range for CHARACTER(KIND=%d)"_warn_en_US,
                  parser::ToUpperCaseLetters(name),
                  static_cast<std::intmax_t>(i.ToInt64()), KIND);
            }
          }
          return CharacterUtils<KIND>::CHAR(i.ToUInt64());
        }));
  } else if (name == "adjustl") {
    return FoldElementalIntrinsic<T, T>(
        context, std::move(funcRef), CharacterUtils<KIND>::ADJUSTL);
  } else if (name == "adjustr") {
    return FoldElementalIntrinsic<T, T>(
        context, std::move(funcRef), CharacterUtils<KIND>::ADJUSTR);
  } else if (name == "max") {
    return FoldMINorMAX(context, std::move(funcRef), Ordering::Greater);
  } else if (name == "maxval") {
    SingleCharType least{0};
    if (auto identity{Identity<T>(
            StringType{least}, GetConstantLength(context, funcRef, 0))}) {
      return FoldMaxvalMinval<T>(
          context, std::move(funcRef), RelationalOperator::GT, *identity);
    }
  } else if (name == "min") {
    return FoldMINorMAX(context, std::move(funcRef), Ordering::Less);
  } else if (name == "minval") {
    // Collating sequences correspond to positive integers (3.31)
    auto most{static_cast<SingleCharType>(0xffffffff >> (8 * (4 - KIND)))};
    if (auto identity{Identity<T>(
            StringType{most}, GetConstantLength(context, funcRef, 0))}) {
      return FoldMaxvalMinval<T>(
          context, std::move(funcRef), RelationalOperator::LT, *identity);
    }
  } else if (name == "new_line") {
    return Expr<T>{Constant<T>{CharacterUtils<KIND>::NEW_LINE()}};
  } else if (name == "repeat") { // not elemental
    if (auto scalars{GetScalarConstantArguments<T, SubscriptInteger>(
            context, funcRef.arguments(), /*hasOptionalArgument=*/false)}) {
      auto str{std::get<Scalar<T>>(*scalars)};
      auto n{std::get<Scalar<SubscriptInteger>>(*scalars).ToInt64()};
      if (n < 0) {
        context.messages().Say(
            "NCOPIES= argument to REPEAT() should be nonnegative, but is %jd"_err_en_US,
            static_cast<std::intmax_t>(n));
      } else if (static_cast<double>(n) * str.size() >
          (1 << 20)) { // sanity limit of 1MiB
        if (context.languageFeatures().ShouldWarn(
                common::UsageWarning::FoldingLimit)) {
          context.messages().Say(common::UsageWarning::FoldingLimit,
              "Result of REPEAT() is too large to compute at compilation time (%g characters)"_port_en_US,
              static_cast<double>(n) * str.size());
        }
      } else {
        return Expr<T>{Constant<T>{CharacterUtils<KIND>::REPEAT(str, n)}};
      }
    }
  } else if (name == "trim") { // not elemental
    if (auto scalar{GetScalarConstantArguments<T>(
            context, funcRef.arguments(), /*hasOptionalArgument=*/false)}) {
      return Expr<T>{Constant<T>{
          CharacterUtils<KIND>::TRIM(std::get<Scalar<T>>(*scalar))}};
    }
  } else if (name == "__builtin_compiler_options") {
    auto &o = context.targetCharacteristics().compilerOptionsString();
    return Expr<T>{Constant<T>{StringType(o.begin(), o.end())}};
  } else if (name == "__builtin_compiler_version") {
    auto &v = context.targetCharacteristics().compilerVersionString();
    return Expr<T>{Constant<T>{StringType(v.begin(), v.end())}};
  }
  return Expr<T>{std::move(funcRef)};
}

template <int KIND>
Expr<Type<TypeCategory::Character, KIND>> FoldOperation(
    FoldingContext &context, Concat<KIND> &&x) {
  if (auto array{ApplyElementwise(context, x)}) {
    return *array;
  }
  using Result = Type<TypeCategory::Character, KIND>;
  if (auto folded{OperandsAreConstants(x)}) {
    return Expr<Result>{Constant<Result>{folded->first + folded->second}};
  }
  return Expr<Result>{std::move(x)};
}

template <int KIND>
Expr<Type<TypeCategory::Character, KIND>> FoldOperation(
    FoldingContext &context, SetLength<KIND> &&x) {
  if (auto array{ApplyElementwise(context, x)}) {
    return *array;
  }
  using Result = Type<TypeCategory::Character, KIND>;
  if (auto folded{OperandsAreConstants(x)}) {
    auto oldLength{static_cast<ConstantSubscript>(folded->first.size())};
    auto newLength{folded->second.ToInt64()};
    if (newLength < oldLength) {
      folded->first.erase(newLength);
    } else {
      folded->first.append(newLength - oldLength, ' ');
    }
    CHECK(static_cast<ConstantSubscript>(folded->first.size()) == newLength);
    return Expr<Result>{Constant<Result>{std::move(folded->first)}};
  }
  return Expr<Result>{std::move(x)};
}

#ifdef _MSC_VER // disable bogus warning about missing definitions
#pragma warning(disable : 4661)
#endif
FOR_EACH_CHARACTER_KIND(template class ExpressionBase, )
template class ExpressionBase<SomeCharacter>;
} // namespace Fortran::evaluate
