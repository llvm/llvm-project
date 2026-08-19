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
    return CharacterUtils::REPEAT(str, std::max<ConstantSubscript>(*len, 0));
  } else {
    return std::nullopt;
  }
}

Expr<Type<TypeCategory::Character>> FoldIntrinsicFunction(
    FoldingContext &context,
    FunctionRef<Type<TypeCategory::Character>> &&funcRef) {
  using T = Type<TypeCategory::Character>;
  using StringType = Scalar<T>; // CharacterValue
  const KindsEnum kind{funcRef.kind()};
  auto *intrinsic{std::get_if<SpecificIntrinsic>(&funcRef.proc().u)};
  CHECK(intrinsic);
  std::string name{intrinsic->name};
  if (name == "achar" || name == "char") {
    using IntT = SubscriptInteger;
    return FoldElementalIntrinsic<T, IntT>(kind, {SubscriptIntegerKind},
        context, std::move(funcRef),
        ScalarFunc<T, IntT>([&](const Scalar<IntT> &i) {
          if (i.IsNegative() ||
              i.BGE(Scalar<IntT>{SubscriptIntegerKind, 0}.IBSET(
                  8 * static_cast<int>(kind)))) {
            context.Warn(common::UsageWarning::FoldingValueChecks,
                "%s(I=%jd) is out of range for CHARACTER(KIND=%d)"_warn_en_US,
                parser::ToUpperCaseLetters(name),
                static_cast<std::intmax_t>(i.ToInt64()),
                static_cast<int>(kind));
          }
          return CharacterUtils::CHAR(static_cast<int>(kind), i.ToUInt64());
        }));
  } else if (name == "adjustl") {
    return FoldElementalIntrinsic<T, T>(
        kind, {kind}, context, std::move(funcRef), CharacterUtils::ADJUSTL);
  } else if (name == "adjustr") {
    return FoldElementalIntrinsic<T, T>(
        kind, {kind}, context, std::move(funcRef), CharacterUtils::ADJUSTR);
  } else if (name == "max") {
    return FoldMINorMAX(context, std::move(funcRef), Ordering::Greater);
  } else if (name == "maxval") {
    StringType least{kind, 1, '\0'};
    if (auto identity{
            Identity<T>(least, GetConstantLength(context, funcRef, 0))}) {
      return FoldMaxvalMinval<T>(
          kind, context, std::move(funcRef), RelationalOperator::GT, *identity);
    }
  } else if (name == "min") {
    return FoldMINorMAX(context, std::move(funcRef), Ordering::Less);
  } else if (name == "minval") {
    // Collating sequences correspond to positive integers (3.31)
    StringType most{kind, 1, 0xffffffff >> (8 * (4 - static_cast<int>(kind)))};
    if (auto identity{
            Identity<T>(most, GetConstantLength(context, funcRef, 0))}) {
      return FoldMaxvalMinval<T>(
          kind, context, std::move(funcRef), RelationalOperator::LT, *identity);
    }
  } else if (name == "new_line") {
    return MakeConstantExpr<T>(
        kind, CharacterUtils::NEW_LINE(static_cast<int>(kind)));
  } else if (name == "repeat") { // not elemental
    if (auto scalars{GetScalarConstantArguments<T, SubscriptInteger>(
            {kind, SubscriptIntegerKind}, context, funcRef.arguments(),
            /*hasOptionalArgument=*/false)}) {
      auto str{std::get<Scalar<T>>(*scalars)};
      auto n{std::get<Scalar<SubscriptInteger>>(*scalars).ToInt64()};
      if (n < 0) {
        context.messages().Say(
            "NCOPIES= argument to REPEAT() should be nonnegative, but is %jd"_err_en_US,
            static_cast<std::intmax_t>(n));
      } else if (static_cast<double>(n) * str.size() >
          (1 << 20)) { // sanity limit of 1MiB
        context.Warn(common::UsageWarning::FoldingLimit,
            "Result of REPEAT() is too large to compute at compilation time (%g characters)"_port_en_US,
            static_cast<double>(n) * str.size());
      } else {
        return MakeConstantExpr<T>(kind, CharacterUtils::REPEAT(str, n));
      }
    }
  } else if (name == "trim") { // not elemental
    if (auto scalar{GetScalarConstantArguments<T>({kind}, context,
            funcRef.arguments(), /*hasOptionalArgument=*/false)}) {
      return MakeConstantExpr<T>(
          kind, CharacterUtils::TRIM(std::get<Scalar<T>>(*scalar)));
    }
  } else if (name == "__builtin_compiler_options") {
    auto &o = context.targetCharacteristics().compilerOptionsString();
    return MakeConstantExpr<T>(kind, o);
  } else if (name == "__builtin_compiler_version") {
    auto &v = context.targetCharacteristics().compilerVersionString();
    return MakeConstantExpr<T>(kind, v);
  }
  return Expr<T>{std::move(funcRef)};
}

Expr<Type<TypeCategory::Character>> FoldOperation(
    FoldingContext &context, Concat &&x) {
  const KindsEnum kind{static_cast<KindsEnum>(x.kind())};
  if (auto array{ApplyElementwise(context, x)}) {
    return *array;
  }
  using Result = Type<TypeCategory::Character>;
  if (auto folded{OperandsAreConstants(x)}) {
    return MakeConstantExpr<Result>(kind, folded->first + folded->second);
  }
  return Expr<Result>{std::move(x)};
}

Expr<Type<TypeCategory::Character>> FoldOperation(
    FoldingContext &context, SetLength &&x) {
  const KindsEnum kind{static_cast<KindsEnum>(x.kind())};
  if (auto array{ApplyElementwise(context, x)}) {
    return *array;
  }
  using Result = Type<TypeCategory::Character>;
  if (auto folded{OperandsAreConstants(x)}) {
    auto oldLength{static_cast<ConstantSubscript>(folded->first.size())};
    auto newLength{folded->second.ToInt64()};
    if (newLength < oldLength) {
      folded->first.erase(newLength);
    } else {
      folded->first.append(newLength - oldLength, ' ');
    }
    CHECK(static_cast<ConstantSubscript>(folded->first.size()) == newLength);
    return MakeConstantExpr<Result>(kind, std::move(folded->first));
  }
  return Expr<Result>{std::move(x)};
}

#ifdef _MSC_VER // disable bogus warning about missing definitions
#pragma warning(disable : 4661)
#endif
FOR_EACH_CHARACTER_KIND(template class ExpressionBase, )
template class ExpressionBase<SomeCharacter>;
} // namespace Fortran::evaluate
