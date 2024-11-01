//===-- lib/Evaluate/fold-real.cpp ----------------------------------------===//
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

template <typename T>
static Expr<T> FoldTransformationalBessel(
    FunctionRef<T> &&funcRef, FoldingContext &context) {
  CHECK(funcRef.arguments().size() == 3);
  /// Bessel runtime functions use `int` integer arguments. Convert integer
  /// arguments to Int4, any overflow error will be reported during the
  /// conversion folding.
  using Int4 = Type<TypeCategory::Integer, 4>;
  if (auto args{GetConstantArguments<Int4, Int4, T>(
          context, funcRef.arguments(), /*hasOptionalArgument=*/false)}) {
    const std::string &name{std::get<SpecificIntrinsic>(funcRef.proc().u).name};
    if (auto elementalBessel{GetHostRuntimeWrapper<T, Int4, T>(name)}) {
      std::vector<Scalar<T>> results;
      int n1{static_cast<int>(
          std::get<0>(*args)->GetScalarValue().value().ToInt64())};
      int n2{static_cast<int>(
          std::get<1>(*args)->GetScalarValue().value().ToInt64())};
      Scalar<T> x{std::get<2>(*args)->GetScalarValue().value()};
      for (int i{n1}; i <= n2; ++i) {
        results.emplace_back((*elementalBessel)(context, Scalar<Int4>{i}, x));
      }
      return Expr<T>{Constant<T>{
          std::move(results), ConstantSubscripts{std::max(n2 - n1 + 1, 0)}}};
    } else if (context.languageFeatures().ShouldWarn(
                   common::UsageWarning::FoldingFailure)) {
      context.messages().Say(common::UsageWarning::FoldingFailure,
          "%s(integer(kind=4), real(kind=%d)) cannot be folded on host"_warn_en_US,
          name, T::kind);
    }
  }
  return Expr<T>{std::move(funcRef)};
}

// NORM2
template <int KIND> class Norm2Accumulator {
  using T = Type<TypeCategory::Real, KIND>;

public:
  Norm2Accumulator(
      const Constant<T> &array, const Constant<T> &maxAbs, Rounding rounding)
      : array_{array}, maxAbs_{maxAbs}, rounding_{rounding} {};
  void operator()(
      Scalar<T> &element, const ConstantSubscripts &at, bool /*first*/) {
    // Summation of scaled elements:
    // Naively,
    //   NORM2(A(:)) = SQRT(SUM(A(:)**2))
    // For any T > 0, we have mathematically
    //   SQRT(SUM(A(:)**2))
    //     = SQRT(T**2 * (SUM(A(:)**2) / T**2))
    //     = SQRT(T**2 * SUM(A(:)**2 / T**2))
    //     = SQRT(T**2 * SUM((A(:)/T)**2))
    //     = SQRT(T**2) * SQRT(SUM((A(:)/T)**2))
    //     = T * SQRT(SUM((A(:)/T)**2))
    // By letting T = MAXVAL(ABS(A)), we ensure that
    // ALL(ABS(A(:)/T) <= 1), so ALL((A(:)/T)**2 <= 1), and the SUM will
    // not overflow unless absolutely necessary.
    auto scale{maxAbs_.At(maxAbsAt_)};
    if (scale.IsZero()) {
      // Maximum value is zero, and so will the result be.
      // Avoid division by zero below.
      element = scale;
    } else {
      auto item{array_.At(at)};
      auto scaled{item.Divide(scale).value};
      auto square{scaled.Multiply(scaled).value};
      if constexpr (useKahanSummation) {
        auto next{square.Add(correction_, rounding_)};
        overflow_ |= next.flags.test(RealFlag::Overflow);
        auto sum{element.Add(next.value, rounding_)};
        overflow_ |= sum.flags.test(RealFlag::Overflow);
        correction_ = sum.value.Subtract(element, rounding_)
                          .value.Subtract(next.value, rounding_)
                          .value;
        element = sum.value;
      } else {
        auto sum{element.Add(square, rounding_)};
        overflow_ |= sum.flags.test(RealFlag::Overflow);
        element = sum.value;
      }
    }
  }
  bool overflow() const { return overflow_; }
  void Done(Scalar<T> &result) {
    // incoming result = SUM((data(:)/maxAbs)**2)
    // outgoing result = maxAbs * SQRT(result)
    auto root{result.SQRT().value};
    auto product{root.Multiply(maxAbs_.At(maxAbsAt_))};
    maxAbs_.IncrementSubscripts(maxAbsAt_);
    overflow_ |= product.flags.test(RealFlag::Overflow);
    result = product.value;
  }

private:
  const Constant<T> &array_;
  const Constant<T> &maxAbs_;
  const Rounding rounding_;
  bool overflow_{false};
  Scalar<T> correction_{};
  ConstantSubscripts maxAbsAt_{maxAbs_.lbounds()};
};

template <int KIND>
static Expr<Type<TypeCategory::Real, KIND>> FoldNorm2(FoldingContext &context,
    FunctionRef<Type<TypeCategory::Real, KIND>> &&funcRef) {
  using T = Type<TypeCategory::Real, KIND>;
  using Element = typename Constant<T>::Element;
  std::optional<int> dim;
  if (std::optional<ArrayAndMask<T>> arrayAndMask{
          ProcessReductionArgs<T>(context, funcRef.arguments(), dim,
              /*X=*/0, /*DIM=*/1)}) {
    MaxvalMinvalAccumulator<T, /*ABS=*/true> maxAbsAccumulator{
        RelationalOperator::GT, context, arrayAndMask->array};
    const Element identity{};
    Constant<T> maxAbs{DoReduction<T>(arrayAndMask->array, arrayAndMask->mask,
        dim, identity, maxAbsAccumulator)};
    Norm2Accumulator norm2Accumulator{arrayAndMask->array, maxAbs,
        context.targetCharacteristics().roundingMode()};
    Constant<T> result{DoReduction<T>(arrayAndMask->array, arrayAndMask->mask,
        dim, identity, norm2Accumulator)};
    if (norm2Accumulator.overflow() &&
        context.languageFeatures().ShouldWarn(
            common::UsageWarning::FoldingException)) {
      context.messages().Say(common::UsageWarning::FoldingException,
          "NORM2() of REAL(%d) data overflowed"_warn_en_US, KIND);
    }
    return Expr<T>{std::move(result)};
  }
  return Expr<T>{std::move(funcRef)};
}

template <int KIND>
Expr<Type<TypeCategory::Real, KIND>> FoldIntrinsicFunction(
    FoldingContext &context,
    FunctionRef<Type<TypeCategory::Real, KIND>> &&funcRef) {
  using T = Type<TypeCategory::Real, KIND>;
  using ComplexT = Type<TypeCategory::Complex, KIND>;
  using Int4 = Type<TypeCategory::Integer, 4>;
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
      name == "log_gamma" || name == "sin" || name == "sinh" || name == "tan" ||
      name == "tanh") {
    CHECK(args.size() == 1);
    if (auto callable{GetHostRuntimeWrapper<T, T>(name)}) {
      return FoldElementalIntrinsic<T, T>(
          context, std::move(funcRef), *callable);
    } else if (context.languageFeatures().ShouldWarn(
                   common::UsageWarning::FoldingFailure)) {
      context.messages().Say(common::UsageWarning::FoldingFailure,
          "%s(real(kind=%d)) cannot be folded on host"_warn_en_US, name, KIND);
    }
  } else if (name == "amax0" || name == "amin0" || name == "amin1" ||
      name == "amax1" || name == "dmin1" || name == "dmax1") {
    return RewriteSpecificMINorMAX(context, std::move(funcRef));
  } else if (name == "atan" || name == "atan2") {
    std::string localName{name == "atan" ? "atan2" : name};
    CHECK(args.size() == 2);
    if (auto callable{GetHostRuntimeWrapper<T, T, T>(localName)}) {
      return FoldElementalIntrinsic<T, T, T>(
          context, std::move(funcRef), *callable);
    } else if (context.languageFeatures().ShouldWarn(
                   common::UsageWarning::FoldingFailure)) {
      context.messages().Say(common::UsageWarning::FoldingFailure,
          "%s(real(kind=%d), real(kind%d)) cannot be folded on host"_warn_en_US,
          name, KIND, KIND);
    }
  } else if (name == "bessel_jn" || name == "bessel_yn") {
    if (args.size() == 2) { // elemental
      // runtime functions use int arg
      if (auto callable{GetHostRuntimeWrapper<T, Int4, T>(name)}) {
        return FoldElementalIntrinsic<T, Int4, T>(
            context, std::move(funcRef), *callable);
      } else if (context.languageFeatures().ShouldWarn(
                     common::UsageWarning::FoldingFailure)) {
        context.messages().Say(common::UsageWarning::FoldingFailure,
            "%s(integer(kind=4), real(kind=%d)) cannot be folded on host"_warn_en_US,
            name, KIND);
      }
    } else {
      return FoldTransformationalBessel<T>(std::move(funcRef), context);
    }
  } else if (name == "abs") { // incl. zabs & cdabs
    // Argument can be complex or real
    if (UnwrapExpr<Expr<SomeReal>>(args[0])) {
      return FoldElementalIntrinsic<T, T>(
          context, std::move(funcRef), &Scalar<T>::ABS);
    } else if (UnwrapExpr<Expr<SomeComplex>>(args[0])) {
      return FoldElementalIntrinsic<T, ComplexT>(context, std::move(funcRef),
          ScalarFunc<T, ComplexT>([&name, &context](
                                      const Scalar<ComplexT> &z) -> Scalar<T> {
            ValueWithRealFlags<Scalar<T>> y{z.ABS()};
            if (y.flags.test(RealFlag::Overflow) &&
                context.languageFeatures().ShouldWarn(
                    common::UsageWarning::FoldingException)) {
              context.messages().Say(common::UsageWarning::FoldingException,
                  "complex ABS intrinsic folding overflow"_warn_en_US, name);
            }
            return y.value;
          }));
    } else {
      common::die(" unexpected argument type inside abs");
    }
  } else if (name == "aimag") {
    if (auto *zExpr{UnwrapExpr<Expr<ComplexT>>(args[0])}) {
      return Fold(context, Expr<T>{ComplexComponent{true, std::move(*zExpr)}});
    }
  } else if (name == "aint" || name == "anint") {
    // ANINT rounds ties away from zero, not to even
    common::RoundingMode mode{name == "aint"
            ? common::RoundingMode::ToZero
            : common::RoundingMode::TiesAwayFromZero};
    return FoldElementalIntrinsic<T, T>(context, std::move(funcRef),
        ScalarFunc<T, T>(
            [&name, &context, mode](const Scalar<T> &x) -> Scalar<T> {
              ValueWithRealFlags<Scalar<T>> y{x.ToWholeNumber(mode)};
              if (y.flags.test(RealFlag::Overflow) &&
                  context.languageFeatures().ShouldWarn(
                      common::UsageWarning::FoldingException)) {
                context.messages().Say(common::UsageWarning::FoldingException,
                    "%s intrinsic folding overflow"_warn_en_US, name);
              }
              return y.value;
            }));
  } else if (name == "dim") {
    return FoldElementalIntrinsic<T, T, T>(context, std::move(funcRef),
        ScalarFunc<T, T, T>([&context](const Scalar<T> &x,
                                const Scalar<T> &y) -> Scalar<T> {
          ValueWithRealFlags<Scalar<T>> result{x.DIM(y)};
          if (result.flags.test(RealFlag::Overflow) &&
              context.languageFeatures().ShouldWarn(
                  common::UsageWarning::FoldingException)) {
            context.messages().Say(common::UsageWarning::FoldingException,
                "DIM intrinsic folding overflow"_warn_en_US);
          }
          return result.value;
        }));
  } else if (name == "dot_product") {
    return FoldDotProduct<T>(context, std::move(funcRef));
  } else if (name == "dprod") {
    // Rewrite DPROD(x,y) -> DBLE(x)*DBLE(y)
    if (args.at(0) && args.at(1)) {
      const auto *xExpr{args[0]->UnwrapExpr()};
      const auto *yExpr{args[1]->UnwrapExpr()};
      if (xExpr && yExpr) {
        return Fold(context,
            ToReal<T::kind>(context, common::Clone(*xExpr)) *
                ToReal<T::kind>(context, common::Clone(*yExpr)));
      }
    }
  } else if (name == "epsilon") {
    return Expr<T>{Scalar<T>::EPSILON()};
  } else if (name == "fraction") {
    return FoldElementalIntrinsic<T, T>(context, std::move(funcRef),
        ScalarFunc<T, T>(
            [](const Scalar<T> &x) -> Scalar<T> { return x.FRACTION(); }));
  } else if (name == "huge") {
    return Expr<T>{Scalar<T>::HUGE()};
  } else if (name == "hypot") {
    CHECK(args.size() == 2);
    return FoldElementalIntrinsic<T, T, T>(context, std::move(funcRef),
        ScalarFunc<T, T, T>(
            [&](const Scalar<T> &x, const Scalar<T> &y) -> Scalar<T> {
              ValueWithRealFlags<Scalar<T>> result{x.HYPOT(y)};
              if (result.flags.test(RealFlag::Overflow) &&
                  context.languageFeatures().ShouldWarn(
                      common::UsageWarning::FoldingException)) {
                context.messages().Say(common::UsageWarning::FoldingException,
                    "HYPOT intrinsic folding overflow"_warn_en_US);
              }
              return result.value;
            }));
  } else if (name == "matmul") {
    return FoldMatmul(context, std::move(funcRef));
  } else if (name == "max") {
    return FoldMINorMAX(context, std::move(funcRef), Ordering::Greater);
  } else if (name == "maxval") {
    return FoldMaxvalMinval<T>(context, std::move(funcRef),
        RelationalOperator::GT, T::Scalar::HUGE().Negate());
  } else if (name == "min") {
    return FoldMINorMAX(context, std::move(funcRef), Ordering::Less);
  } else if (name == "minval") {
    return FoldMaxvalMinval<T>(
        context, std::move(funcRef), RelationalOperator::LT, T::Scalar::HUGE());
  } else if (name == "mod") {
    CHECK(args.size() == 2);
    bool badPConst{false};
    if (auto *pExpr{UnwrapExpr<Expr<T>>(args[1])}) {
      *pExpr = Fold(context, std::move(*pExpr));
      if (auto pConst{GetScalarConstantValue<T>(*pExpr)}; pConst &&
          pConst->IsZero() &&
          context.languageFeatures().ShouldWarn(
              common::UsageWarning::FoldingAvoidsRuntimeCrash)) {
        context.messages().Say(common::UsageWarning::FoldingAvoidsRuntimeCrash,
            "MOD: P argument is zero"_warn_en_US);
        badPConst = true;
      }
    }
    return FoldElementalIntrinsic<T, T, T>(context, std::move(funcRef),
        ScalarFunc<T, T, T>([&context, badPConst](const Scalar<T> &x,
                                const Scalar<T> &y) -> Scalar<T> {
          auto result{x.MOD(y)};
          if (!badPConst && result.flags.test(RealFlag::DivideByZero) &&
              context.languageFeatures().ShouldWarn(
                  common::UsageWarning::FoldingAvoidsRuntimeCrash)) {
            context.messages().Say(
                common::UsageWarning::FoldingAvoidsRuntimeCrash,
                "second argument to MOD must not be zero"_warn_en_US);
          }
          return result.value;
        }));
  } else if (name == "modulo") {
    CHECK(args.size() == 2);
    bool badPConst{false};
    if (auto *pExpr{UnwrapExpr<Expr<T>>(args[1])}) {
      *pExpr = Fold(context, std::move(*pExpr));
      if (auto pConst{GetScalarConstantValue<T>(*pExpr)}; pConst &&
          pConst->IsZero() &&
          context.languageFeatures().ShouldWarn(
              common::UsageWarning::FoldingAvoidsRuntimeCrash)) {
        context.messages().Say(common::UsageWarning::FoldingAvoidsRuntimeCrash,
            "MODULO: P argument is zero"_warn_en_US);
        badPConst = true;
      }
    }
    return FoldElementalIntrinsic<T, T, T>(context, std::move(funcRef),
        ScalarFunc<T, T, T>([&context, badPConst](const Scalar<T> &x,
                                const Scalar<T> &y) -> Scalar<T> {
          auto result{x.MODULO(y)};
          if (!badPConst && result.flags.test(RealFlag::DivideByZero) &&
              context.languageFeatures().ShouldWarn(
                  common::UsageWarning::FoldingAvoidsRuntimeCrash)) {
            context.messages().Say(
                common::UsageWarning::FoldingAvoidsRuntimeCrash,
                "second argument to MODULO must not be zero"_warn_en_US);
          }
          return result.value;
        }));
  } else if (name == "nearest") {
    if (auto *sExpr{UnwrapExpr<Expr<SomeReal>>(args[1])}) {
      *sExpr = Fold(context, std::move(*sExpr));
      return common::visit(
          [&](const auto &sVal) {
            using TS = ResultType<decltype(sVal)>;
            bool badSConst{false};
            if (auto sConst{GetScalarConstantValue<TS>(sVal)}; sConst &&
                (sConst->IsZero() || sConst->IsNotANumber()) &&
                context.languageFeatures().ShouldWarn(
                    common::UsageWarning::FoldingValueChecks)) {
              context.messages().Say(common::UsageWarning::FoldingValueChecks,
                  "NEAREST: S argument is %s"_warn_en_US,
                  sConst->IsZero() ? "zero" : "NaN");
              badSConst = true;
            }
            return FoldElementalIntrinsic<T, T, TS>(context, std::move(funcRef),
                ScalarFunc<T, T, TS>([&](const Scalar<T> &x,
                                         const Scalar<TS> &s) -> Scalar<T> {
                  if (!badSConst && (s.IsZero() || s.IsNotANumber()) &&
                      context.languageFeatures().ShouldWarn(
                          common::UsageWarning::FoldingValueChecks)) {
                    context.messages().Say(
                        common::UsageWarning::FoldingValueChecks,
                        "NEAREST: S argument is %s"_warn_en_US,
                        s.IsZero() ? "zero" : "NaN");
                  }
                  auto result{x.NEAREST(!s.IsNegative())};
                  if (context.languageFeatures().ShouldWarn(
                          common::UsageWarning::FoldingException)) {
                    if (result.flags.test(RealFlag::InvalidArgument)) {
                      context.messages().Say(
                          common::UsageWarning::FoldingException,
                          "NEAREST intrinsic folding: bad argument"_warn_en_US);
                    }
                  }
                  return result.value;
                }));
          },
          sExpr->u);
    }
  } else if (name == "norm2") {
    return FoldNorm2<T::kind>(context, std::move(funcRef));
  } else if (name == "product") {
    auto one{Scalar<T>::FromInteger(value::Integer<8>{1}).value};
    return FoldProduct<T>(context, std::move(funcRef), one);
  } else if (name == "real" || name == "dble") {
    if (auto *expr{args[0].value().UnwrapExpr()}) {
      return ToReal<KIND>(context, std::move(*expr));
    }
  } else if (name == "rrspacing") {
    return FoldElementalIntrinsic<T, T>(context, std::move(funcRef),
        ScalarFunc<T, T>(
            [](const Scalar<T> &x) -> Scalar<T> { return x.RRSPACING(); }));
  } else if (name == "scale") {
    if (const auto *byExpr{UnwrapExpr<Expr<SomeInteger>>(args[1])}) {
      return common::visit(
          [&](const auto &byVal) {
            using TBY = ResultType<decltype(byVal)>;
            return FoldElementalIntrinsic<T, T, TBY>(context,
                std::move(funcRef),
                ScalarFunc<T, T, TBY>(
                    [&](const Scalar<T> &x, const Scalar<TBY> &y) -> Scalar<T> {
                      ValueWithRealFlags<Scalar<T>> result{
                          x.
// MSVC chokes on the keyword "template" here in a call to a
// member function template.
#ifndef _MSC_VER
                          template
#endif
                          SCALE<Scalar<TBY>>(y)};
                      if (result.flags.test(RealFlag::Overflow) &&
                          context.languageFeatures().ShouldWarn(
                              common::UsageWarning::FoldingException)) {
                        context.messages().Say(
                            common::UsageWarning::FoldingException,
                            "SCALE intrinsic folding overflow"_warn_en_US);
                      }
                      return result.value;
                    }));
          },
          byExpr->u);
    }
  } else if (name == "set_exponent") {
    if (const auto *iExpr{UnwrapExpr<Expr<SomeInteger>>(args[1])}) {
      return common::visit(
          [&](const auto &iVal) {
            using TY = ResultType<decltype(iVal)>;
            return FoldElementalIntrinsic<T, T, TY>(context, std::move(funcRef),
                ScalarFunc<T, T, TY>(
                    [&](const Scalar<T> &x, const Scalar<TY> &i) -> Scalar<T> {
                      return x.SET_EXPONENT(i.ToInt64());
                    }));
          },
          iExpr->u);
    }
  } else if (name == "sign") {
    return FoldElementalIntrinsic<T, T, T>(
        context, std::move(funcRef), &Scalar<T>::SIGN);
  } else if (name == "spacing") {
    return FoldElementalIntrinsic<T, T>(context, std::move(funcRef),
        ScalarFunc<T, T>(
            [](const Scalar<T> &x) -> Scalar<T> { return x.SPACING(); }));
  } else if (name == "sqrt") {
    return FoldElementalIntrinsic<T, T>(context, std::move(funcRef),
        ScalarFunc<T, T>(
            [](const Scalar<T> &x) -> Scalar<T> { return x.SQRT().value; }));
  } else if (name == "sum") {
    return FoldSum<T>(context, std::move(funcRef));
  } else if (name == "tiny") {
    return Expr<T>{Scalar<T>::TINY()};
  } else if (name == "__builtin_fma") {
    CHECK(args.size() == 3);
  } else if (name == "__builtin_ieee_next_after") {
    if (const auto *yExpr{UnwrapExpr<Expr<SomeReal>>(args[1])}) {
      return common::visit(
          [&](const auto &yVal) {
            using TY = ResultType<decltype(yVal)>;
            return FoldElementalIntrinsic<T, T, TY>(context, std::move(funcRef),
                ScalarFunc<T, T, TY>([&](const Scalar<T> &x,
                                         const Scalar<TY> &y) -> Scalar<T> {
                  auto xBig{Scalar<LargestReal>::Convert(x).value};
                  auto yBig{Scalar<LargestReal>::Convert(y).value};
                  switch (xBig.Compare(yBig)) {
                  case Relation::Unordered:
                    if (context.languageFeatures().ShouldWarn(
                            common::UsageWarning::FoldingValueChecks)) {
                      context.messages().Say(
                          common::UsageWarning::FoldingValueChecks,
                          "IEEE_NEXT_AFTER intrinsic folding: arguments are unordered"_warn_en_US);
                    }
                    return x.NotANumber();
                  case Relation::Equal:
                    break;
                  case Relation::Less:
                    return x.NEAREST(true).value;
                  case Relation::Greater:
                    return x.NEAREST(false).value;
                  }
                  return x; // dodge bogus "missing return" GCC warning
                }));
          },
          yExpr->u);
    }
  } else if (name == "__builtin_ieee_next_up" ||
      name == "__builtin_ieee_next_down") {
    bool upward{name == "__builtin_ieee_next_up"};
    const char *iName{upward ? "IEEE_NEXT_UP" : "IEEE_NEXT_DOWN"};
    return FoldElementalIntrinsic<T, T>(context, std::move(funcRef),
        ScalarFunc<T, T>([&](const Scalar<T> &x) -> Scalar<T> {
          auto result{x.NEAREST(upward)};
          if (context.languageFeatures().ShouldWarn(
                  common::UsageWarning::FoldingException)) {
            if (result.flags.test(RealFlag::InvalidArgument)) {
              context.messages().Say(common::UsageWarning::FoldingException,
                  "%s intrinsic folding: argument is NaN"_warn_en_US, iName);
            }
          }
          return result.value;
        }));
  }
  return Expr<T>{std::move(funcRef)};
}

#ifdef _MSC_VER // disable bogus warning about missing definitions
#pragma warning(disable : 4661)
#endif
FOR_EACH_REAL_KIND(template class ExpressionBase, )
template class ExpressionBase<SomeReal>;
} // namespace Fortran::evaluate
