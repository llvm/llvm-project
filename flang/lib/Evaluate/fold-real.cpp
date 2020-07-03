//===-- lib/Evaluate/fold-real.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "fold-implementation.h"

namespace Fortran::evaluate {

template <int KIND>
Expr<Type<TypeCategory::Real, KIND>> FoldIntrinsicFunction(
    FoldingContext &context,
    FunctionRef<Type<TypeCategory::Real, KIND>> &&funcRef) {
  using T = Type<TypeCategory::Real, KIND>;
  using ComplexT = Type<TypeCategory::Complex, KIND>;
  ActualArguments &args{funcRef.arguments()};
  auto *intrinsic{std::get_if<SpecificIntrinsic>(&funcRef.proc().u)};
  CHECK(intrinsic);
  std::string name{intrinsic->name};
  if (name == "acos" || name == "acosh" || name == "asin" || name == "asinh" ||
      (name == "atan" && args.size() == 1) || name == "atanh" ||
      name == "bessel_j0" || name == "bessel_j1" || name == "bessel_y0" ||
      name == "bessel_y1" || name == "cos" || name == "cosh" || name == "erf" ||
      name == "erfc" || name == "erfc_scaled" || name == "exp" ||
      name == "gamma" || name == "log" || name == "log10" ||
      name == "log_gamma" || name == "sin" || name == "sinh" ||
      name == "sqrt" || name == "tan" || name == "tanh") {
    CHECK(args.size() == 1);
    if (auto callable{context.hostIntrinsicsLibrary()
                          .GetHostProcedureWrapper<Scalar, T, T>(name)}) {
      return FoldElementalIntrinsic<T, T>(
          context, std::move(funcRef), *callable);
    } else {
      context.messages().Say(
          "%s(real(kind=%d)) cannot be folded on host"_en_US, name, KIND);
    }
  } else if (name == "amax0" || name == "amin0" || name == "amin1" ||
      name == "amax1" || name == "dmin1" || name == "dmax1") {
    return RewriteSpecificMINorMAX(context, std::move(funcRef));
  } else if (name == "atan" || name == "atan2" || name == "hypot" ||
      name == "mod") {
    std::string localName{name == "atan" ? "atan2" : name};
    CHECK(args.size() == 2);
    if (auto callable{
            context.hostIntrinsicsLibrary()
                .GetHostProcedureWrapper<Scalar, T, T, T>(localName)}) {
      return FoldElementalIntrinsic<T, T, T>(
          context, std::move(funcRef), *callable);
    } else {
      context.messages().Say(
          "%s(real(kind=%d), real(kind%d)) cannot be folded on host"_en_US,
          name, KIND, KIND);
    }
  } else if (name == "bessel_jn" || name == "bessel_yn") {
    if (args.size() == 2) { // elemental
      // runtime functions use int arg
      using Int4 = Type<TypeCategory::Integer, 4>;
      if (auto callable{
              context.hostIntrinsicsLibrary()
                  .GetHostProcedureWrapper<Scalar, T, Int4, T>(name)}) {
        return FoldElementalIntrinsic<T, Int4, T>(
            context, std::move(funcRef), *callable);
      } else {
        context.messages().Say(
            "%s(integer(kind=4), real(kind=%d)) cannot be folded on host"_en_US,
            name, KIND);
      }
    }
  } else if (name == "abs") {
    // Argument can be complex or real
    if (auto *x{UnwrapExpr<Expr<SomeReal>>(args[0])}) {
      return FoldElementalIntrinsic<T, T>(
          context, std::move(funcRef), &Scalar<T>::ABS);
    } else if (auto *z{UnwrapExpr<Expr<SomeComplex>>(args[0])}) {
      if (auto callable{
              context.hostIntrinsicsLibrary()
                  .GetHostProcedureWrapper<Scalar, T, ComplexT>("abs")}) {
        return FoldElementalIntrinsic<T, ComplexT>(
            context, std::move(funcRef), *callable);
      } else {
        context.messages().Say(
            "abs(complex(kind=%d)) cannot be folded on host"_en_US, KIND);
      }
    } else {
      common::die(" unexpected argument type inside abs");
    }
  } else if (name == "aimag") {
    return FoldElementalIntrinsic<T, ComplexT>(
        context, std::move(funcRef), &Scalar<ComplexT>::AIMAG);
  } else if (name == "aint" || name == "anint") {
    // ANINT rounds ties away from zero, not to even
    common::RoundingMode mode{name == "aint"
            ? common::RoundingMode::ToZero
            : common::RoundingMode::TiesAwayFromZero};
    return FoldElementalIntrinsic<T, T>(context, std::move(funcRef),
        ScalarFunc<T, T>([&name, &context, mode](
                             const Scalar<T> &x) -> Scalar<T> {
          ValueWithRealFlags<Scalar<T>> y{x.ToWholeNumber(mode)};
          if (y.flags.test(RealFlag::Overflow)) {
            context.messages().Say("%s intrinsic folding overflow"_en_US, name);
          }
          return y.value;
        }));
  } else if (name == "dprod") {
    if (auto scalars{GetScalarConstantArguments<T, T>(context, args)}) {
      return Fold(context,
          Expr<T>{Multiply<T>{
              Expr<T>{std::get<0>(*scalars)}, Expr<T>{std::get<1>(*scalars)}}});
    }
  } else if (name == "epsilon") {
    return Expr<T>{Scalar<T>::EPSILON()};
  } else if (name == "huge") {
    return Expr<T>{Scalar<T>::HUGE()};
  } else if (name == "max") {
    return FoldMINorMAX(context, std::move(funcRef), Ordering::Greater);
  } else if (name == "merge") {
    return FoldMerge<T>(context, std::move(funcRef));
  } else if (name == "min") {
    return FoldMINorMAX(context, std::move(funcRef), Ordering::Less);
  } else if (name == "real") {
    if (auto *expr{args[0].value().UnwrapExpr()}) {
      return ToReal<KIND>(context, std::move(*expr));
    }
  } else if (name == "sign") {
    return FoldElementalIntrinsic<T, T, T>(
        context, std::move(funcRef), &Scalar<T>::SIGN);
  } else if (name == "tiny") {
    return Expr<T>{Scalar<T>::TINY()};
  }
  // TODO: cshift, dim, dot_product, eoshift, fraction, matmul,
  // maxval, minval, modulo, nearest, norm2, pack, product,
  // reduce, rrspacing, scale, set_exponent, spacing, spread,
  // sum, transfer, transpose, unpack, bessel_jn (transformational) and
  // bessel_yn (transformational)
  return Expr<T>{std::move(funcRef)};
}

template <int KIND>
Expr<Type<TypeCategory::Real, KIND>> FoldOperation(
    FoldingContext &context, ComplexComponent<KIND> &&x) {
  using Operand = Type<TypeCategory::Complex, KIND>;
  using Result = Type<TypeCategory::Real, KIND>;
  if (auto array{ApplyElementwise(context, x,
          std::function<Expr<Result>(Expr<Operand> &&)>{
              [=](Expr<Operand> &&operand) {
                return Expr<Result>{ComplexComponent<KIND>{
                    x.isImaginaryPart, std::move(operand)}};
              }})}) {
    return *array;
  }
  using Part = Type<TypeCategory::Real, KIND>;
  auto &operand{x.left()};
  if (auto value{GetScalarConstantValue<Operand>(operand)}) {
    if (x.isImaginaryPart) {
      return Expr<Part>{Constant<Part>{value->AIMAG()}};
    } else {
      return Expr<Part>{Constant<Part>{value->REAL()}};
    }
  }
  return Expr<Part>{std::move(x)};
}

FOR_EACH_REAL_KIND(template class ExpressionBase, )
template class ExpressionBase<SomeReal>;
} // namespace Fortran::evaluate
