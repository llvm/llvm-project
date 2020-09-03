//===-- lib/Evaluate/fold-character.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "fold-implementation.h"

namespace Fortran::evaluate {

template <int KIND>
Expr<Type<TypeCategory::Character, KIND>> FoldIntrinsicFunction(
    FoldingContext &context,
    FunctionRef<Type<TypeCategory::Character, KIND>> &&funcRef) {
  using T = Type<TypeCategory::Character, KIND>;
  auto *intrinsic{std::get_if<SpecificIntrinsic>(&funcRef.proc().u)};
  CHECK(intrinsic);
  std::string name{intrinsic->name};
  if (name == "achar" || name == "char") {
    using IntT = SubscriptInteger;
    return FoldElementalIntrinsic<T, IntT>(context, std::move(funcRef),
        ScalarFunc<T, IntT>([](const Scalar<IntT> &i) {
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
  } else if (name == "merge") {
    return FoldMerge<T>(context, std::move(funcRef));
  } else if (name == "min") {
    return FoldMINorMAX(context, std::move(funcRef), Ordering::Less);
  } else if (name == "new_line") {
    return Expr<T>{Constant<T>{CharacterUtils<KIND>::NEW_LINE()}};
  } else if (name == "repeat") { // not elemental
    if (auto scalars{GetScalarConstantArguments<T, SubscriptInteger>(
            context, funcRef.arguments())}) {
      return Expr<T>{Constant<T>{
          CharacterUtils<KIND>::REPEAT(std::get<Scalar<T>>(*scalars),
              std::get<Scalar<SubscriptInteger>>(*scalars).ToInt64())}};
    }
  } else if (name == "trim") { // not elemental
    if (auto scalar{
            GetScalarConstantArguments<T>(context, funcRef.arguments())}) {
      return Expr<T>{Constant<T>{
          CharacterUtils<KIND>::TRIM(std::get<Scalar<T>>(*scalar))}};
    }
  }
  // TODO: cshift, eoshift, maxval, minval, pack, reduce,
  // spread, transfer, transpose, unpack
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

FOR_EACH_CHARACTER_KIND(template class ExpressionBase, )
template class ExpressionBase<SomeCharacter>;
} // namespace Fortran::evaluate
