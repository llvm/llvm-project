//===-- lib/Evaluate/fold-complex.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "fold-implementation.h"
#include "fold-matmul.h"
#include "fold-reduction.h"

namespace Fortran::evaluate {

Expr<Type<TypeCategory::Complex>> FoldIntrinsicFunction(FoldingContext &context,
    FunctionRef<Type<TypeCategory::Complex>> &&funcRef) {
  using T = Type<TypeCategory::Complex>;
  using Part = typename T::Part;
  const int kind{funcRef.GetType().value().kind()};
  ActualArguments &args{funcRef.arguments()};
  auto *intrinsic{std::get_if<SpecificIntrinsic>(&funcRef.proc().u)};
  CHECK(intrinsic);
  std::string name{intrinsic->name};
  if (name == "acos" || name == "acosh" || name == "asin" || name == "asinh" ||
      name == "atan" || name == "atanh" || name == "cos" || name == "cosh" ||
      name == "exp" || name == "log" || name == "sin" || name == "sinh" ||
      name == "sqrt" || name == "tan" || name == "tanh") {
    if (auto callable{GetHostRuntimeWrapper<T, T>(name, kind)}) {
      return FoldElementalIntrinsic<T, T>(
          context, std::move(funcRef), *callable);
    } else {
      context.Warn(common::UsageWarning::FoldingFailure,
          "%s(complex(kind=%d)) cannot be folded on host"_warn_en_US, name,
          kind);
    }
  } else if (name == "conjg") {
    return FoldElementalIntrinsic<T, T>(
        context, std::move(funcRef), &Scalar<T>::CONJG);
  } else if (name == "cmplx") {
    if (args.size() > 0 && args[0].has_value()) {
      if (auto *x{UnwrapExpr<Expr<SomeComplex>>(args[0])}) {
        // CMPLX(X [, KIND]) with complex X
        return Fold(context, ConvertToType<T>(kind, std::move(*x)));
      } else {
        if (args.size() >= 2 && args[1].has_value()) {
          // Do not fold CMPLX with an Y argument that may be absent at runtime
          // into a complex constructor so that lowering can deal with the
          // optional aspect (there is no optional aspect with the complex
          // constructor).
          if (MayBePassedAsAbsentOptional(*args[1]->UnwrapExpr())) {
            return Expr<T>{std::move(funcRef)};
          }
        }
        // CMPLX(X [, Y [, KIND]]) with non-complex X
        Expr<SomeType> re{std::move(*args[0].value().UnwrapExpr())};
        Expr<SomeType> im{args.size() >= 2 && args[1].has_value()
                ? std::move(*args[1]->UnwrapExpr())
                : AsGenericExpr(Constant<Part>{Scalar<Part>::Zero(kind)})};
        return Fold(context,
            Expr<T>{ComplexConstructor{ToReal(context, std::move(re), kind),
                ToReal(context, std::move(im), kind)}});
      }
    }
  } else if (name == "dot_product") {
    return FoldDotProduct<T>(context, std::move(funcRef));
  } else if (name == "matmul") {
    return FoldMatmul(context, std::move(funcRef));
  } else if (name == "product") {
    auto one{Scalar<Part>::FromInteger(value::IntegerValue{1, 1}, kind).value};
    return FoldProduct<T>(context, std::move(funcRef), Scalar<T>{one});
  } else if (name == "sum") {
    return FoldSum<T>(context, std::move(funcRef));
  }
  return Expr<T>{std::move(funcRef)};
}

Expr<Type<TypeCategory::Complex>> FoldOperation(
    FoldingContext &context, ComplexConstructor &&x) {
  if (auto array{ApplyElementwise(context, x)}) {
    return *array;
  }
  using ComplexType = Type<TypeCategory::Complex>;
  if (auto folded{OperandsAreConstants(x)}) {
    using RealType = typename ComplexType::Part;
    Constant<ComplexType> result{
        Scalar<ComplexType>{folded->first, folded->second}};
    if (const auto *re{UnwrapConstantValue<RealType>(x.left())};
        re && re->result().isFromInexactLiteralConversion()) {
      result.result().set_isFromInexactLiteralConversion();
    } else if (const auto *im{UnwrapConstantValue<RealType>(x.right())};
        im && im->result().isFromInexactLiteralConversion()) {
      result.result().set_isFromInexactLiteralConversion();
    }
    return Expr<ComplexType>{std::move(result)};
  }
  return Expr<ComplexType>{std::move(x)};
}

#ifdef _MSC_VER // disable bogus warning about missing definitions
#pragma warning(disable : 4661)
#endif
FOR_EACH_COMPLEX_KIND(template class ExpressionBase, )
template class ExpressionBase<SomeComplex>;
} // namespace Fortran::evaluate
