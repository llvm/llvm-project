//===-- lib/Evaluate/fold-complex.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "fold-implementation.h"

namespace Fortran::evaluate {

template <int KIND>
Expr<Type<TypeCategory::Complex, KIND>> FoldIntrinsicFunction(
    FoldingContext &context,
    FunctionRef<Type<TypeCategory::Complex, KIND>> &&funcRef) {
  using T = Type<TypeCategory::Complex, KIND>;
  ActualArguments &args{funcRef.arguments()};
  auto *intrinsic{std::get_if<SpecificIntrinsic>(&funcRef.proc().u)};
  CHECK(intrinsic);
  std::string name{intrinsic->name};
  if (name == "acos" || name == "acosh" || name == "asin" || name == "asinh" ||
      name == "atan" || name == "atanh" || name == "cos" || name == "cosh" ||
      name == "exp" || name == "log" || name == "sin" || name == "sinh" ||
      name == "sqrt" || name == "tan" || name == "tanh") {
    if (auto callable{GetHostRuntimeWrapper<T, T>(name)}) {
      return FoldElementalIntrinsic<T, T>(
          context, std::move(funcRef), *callable);
    } else {
      context.messages().Say(
          "%s(complex(kind=%d)) cannot be folded on host"_en_US, name, KIND);
    }
  } else if (name == "conjg") {
    return FoldElementalIntrinsic<T, T>(
        context, std::move(funcRef), &Scalar<T>::CONJG);
  } else if (name == "cmplx") {
    if (args.size() > 0 && args[0].has_value()) {
      if (auto *x{UnwrapExpr<Expr<SomeComplex>>(args[0])}) {
        // CMPLX(X [, KIND]) with complex X
        return Fold(context, ConvertToType<T>(std::move(*x)));
      } else {
        // CMPLX(X [, Y [, KIND]]) with non-complex X
        using Part = typename T::Part;
        Expr<SomeType> re{std::move(*args[0].value().UnwrapExpr())};
        Expr<SomeType> im{args.size() >= 2 && args[1].has_value()
                ? std::move(*args[1]->UnwrapExpr())
                : AsGenericExpr(Constant<Part>{Scalar<Part>{}})};
        return Fold(context,
            Expr<T>{
                ComplexConstructor<KIND>{ToReal<KIND>(context, std::move(re)),
                    ToReal<KIND>(context, std::move(im))}});
      }
    }
  } else if (name == "merge") {
    return FoldMerge<T>(context, std::move(funcRef));
  }
  // TODO: cshift, dot_product, eoshift, matmul, pack, product,
  // reduce, spread, sum, transfer, transpose, unpack
  return Expr<T>{std::move(funcRef)};
}

template <int KIND>
Expr<Type<TypeCategory::Complex, KIND>> FoldOperation(
    FoldingContext &context, ComplexConstructor<KIND> &&x) {
  if (auto array{ApplyElementwise(context, x)}) {
    return *array;
  }
  using Result = Type<TypeCategory::Complex, KIND>;
  if (auto folded{OperandsAreConstants(x)}) {
    return Expr<Result>{
        Constant<Result>{Scalar<Result>{folded->first, folded->second}}};
  }
  return Expr<Result>{std::move(x)};
}

FOR_EACH_COMPLEX_KIND(template class ExpressionBase, )
template class ExpressionBase<SomeComplex>;
} // namespace Fortran::evaluate
