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

// Dispatches a compile-time CHARACTER KIND to a runtime kind by invoking
// f(std::integral_constant<int, KIND>{}).  All branches must return the same
// type.
template <typename F> static auto WithCharKind(int kind, F &&f) {
  switch (kind) {
  case 2:
    return f(std::integral_constant<int, 2>{});
  case 4:
    return f(std::integral_constant<int, 4>{});
  default:
    return f(std::integral_constant<int, 1>{});
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
  const int kind{funcRef.GetType().value().kind()};
  auto *intrinsic{std::get_if<SpecificIntrinsic>(&funcRef.proc().u)};
  CHECK(intrinsic);
  std::string name{intrinsic->name};
  if (name == "achar" || name == "char") {
    using IntT = SubscriptInteger;
    // The code argument is folded as a subscript-kind (KIND=8) integer; force
    // that kind here (it is a runtime property now that the type is kindless)
    // so that the range check and CHAR conversion below operate on a
    // full-width value rather than the argument's narrower native kind.
    Folder<IntT>(context, /*forOptionalArgument=*/false,
        /*toKind=*/subscriptIntegerKind)
        .Folding(funcRef.arguments().at(0));
    return FoldElementalIntrinsic<T, IntT>(context, std::move(funcRef),
        ScalarFunc<T, IntT>([&](const Scalar<IntT> &i) {
          if (i.IsNegative() ||
              i.BGE(value::IntegerValue{0, subscriptIntegerKind}.IBSET(
                  8 * kind))) {
            context.Warn(common::UsageWarning::FoldingValueChecks,
                "%s(I=%jd) is out of range for CHARACTER(KIND=%d)"_warn_en_US,
                parser::ToUpperCaseLetters(name),
                static_cast<std::intmax_t>(i.ToInt64()), kind);
          }
          return CharacterUtils::CHAR(i.ToUInt64(), kind);
        }));
  } else if (name == "adjustl") {
    return FoldElementalIntrinsic<T, T>(context, std::move(funcRef),
        ScalarFunc<T, T>(
            [](const Scalar<T> &s) { return CharacterUtils::ADJUSTL(s); }));
  } else if (name == "adjustr") {
    return FoldElementalIntrinsic<T, T>(context, std::move(funcRef),
        ScalarFunc<T, T>(
            [](const Scalar<T> &s) { return CharacterUtils::ADJUSTR(s); }));
  } else if (name == "max") {
    return FoldMINorMAX(context, std::move(funcRef), Ordering::Greater);
  } else if (name == "maxval") {
    StringType least{WithCharKind(kind, [](auto k) {
      using CharT = std::conditional_t<k.value == 1, char,
          std::conditional_t<k.value == 2, char16_t, char32_t>>;
      return StringType(std::size_t{1}, CharT{0});
    })};
    if (auto identity{
            Identity<T>(least, GetConstantLength(context, funcRef, 0))}) {
      return FoldMaxvalMinval<T>(
          context, std::move(funcRef), RelationalOperator::GT, *identity);
    }
  } else if (name == "min") {
    return FoldMINorMAX(context, std::move(funcRef), Ordering::Less);
  } else if (name == "minval") {
    // Collating sequences correspond to positive integers (3.31)
    StringType most{WithCharKind(kind, [](auto k) {
      using CharT = std::conditional_t<k.value == 1, char,
          std::conditional_t<k.value == 2, char16_t, char32_t>>;
      return StringType(std::size_t{1},
          static_cast<CharT>(0xffffffffu >> (8 * (4 - k.value))));
    })};
    if (auto identity{
            Identity<T>(most, GetConstantLength(context, funcRef, 0))}) {
      return FoldMaxvalMinval<T>(
          context, std::move(funcRef), RelationalOperator::LT, *identity);
    }
  } else if (name == "new_line") {
    return Expr<T>{Constant<T>{CharacterUtils::NEW_LINE(kind)}};
  } else if (name == "repeat") { // not elemental
    // NCOPIES is folded as a subscript-kind (KIND=8) integer; force that kind
    // (it is a runtime property now that the type is kindless) so the result
    // length computation LEN(STRING)*NCOPIES uses a full-width value.
    Folder<SubscriptInteger>(context, /*forOptionalArgument=*/false,
        /*toKind=*/subscriptIntegerKind)
        .Folding(funcRef.arguments().at(1));
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
        context.Warn(common::UsageWarning::FoldingLimit,
            "Result of REPEAT() is too large to compute at compilation time (%g characters)"_port_en_US,
            static_cast<double>(n) * str.size());
      } else {
        return Expr<T>{Constant<T>{CharacterUtils::REPEAT(str, n)}};
      }
    }
  } else if (name == "trim") { // not elemental
    if (auto scalar{GetScalarConstantArguments<T>(
            context, funcRef.arguments(), /*hasOptionalArgument=*/false)}) {
      return Expr<T>{
          Constant<T>{CharacterUtils::TRIM(std::get<Scalar<T>>(*scalar))}};
    }
  } else if (name == "__builtin_compiler_options") {
    auto &o = context.targetCharacteristics().compilerOptionsString();
    return Expr<T>{Constant<T>{WithCharKind(kind, [&](auto k) {
      using RawStr = std::conditional_t<k.value == 1, std::string,
          std::conditional_t<k.value == 2, std::u16string, std::u32string>>;
      return StringType{RawStr(o.begin(), o.end())};
    })}};
  } else if (name == "__builtin_compiler_version") {
    auto &v = context.targetCharacteristics().compilerVersionString();
    return Expr<T>{Constant<T>{WithCharKind(kind, [&](auto k) {
      using RawStr = std::conditional_t<k.value == 1, std::string,
          std::conditional_t<k.value == 2, std::u16string, std::u32string>>;
      return StringType{RawStr(v.begin(), v.end())};
    })}};
  }
  return Expr<T>{std::move(funcRef)};
}

Expr<Type<TypeCategory::Character>> FoldOperation(
    FoldingContext &context, Concat &&x) {
  if (auto array{ApplyElementwise(context, x)}) {
    return *array;
  }
  using Result = Type<TypeCategory::Character>;
  if (auto folded{OperandsAreConstants(x)}) {
    return Expr<Result>{Constant<Result>{folded->first + folded->second}};
  }
  return Expr<Result>{std::move(x)};
}

Expr<Type<TypeCategory::Character>> FoldOperation(
    FoldingContext &context, SetLength &&x) {
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
      // append(n, char) widens the fill character to the string's element type.
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
