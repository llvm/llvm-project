#include "../../lib/Evaluate/host.h"
#include "flang/Evaluate/call.h"
#include "flang/Evaluate/expression.h"
#include "flang/Evaluate/fold.h"
#include "flang/Evaluate/intrinsics-library.h"
#include "flang/Evaluate/intrinsics.h"
#include "flang/Evaluate/target.h"
#include "flang/Evaluate/tools.h"
#include "flang/Testing/testing.h"
#include <tuple>

using namespace Fortran::evaluate;

// helper to call functions on all types from tuple
template <typename... T> struct RunOnTypes {};
template <typename Test, typename... T>
struct RunOnTypes<Test, std::tuple<T...>> {
  static void Run() { (..., Test::template Run<T>()); }
};

// test for fold.h GetScalarConstantValue function
struct TestGetScalarConstantValue {
  template <typename T> static void Run() {
    Expr<T> exprFullyTyped{Constant<T>{Scalar<T>{}}};
    Expr<SomeKind<T::category>> exprSomeKind{exprFullyTyped};
    Expr<SomeType> exprSomeType{exprSomeKind};
    TEST(GetScalarConstantValue<T>(exprFullyTyped).has_value());
    TEST(GetScalarConstantValue<T>(exprSomeKind).has_value());
    TEST(GetScalarConstantValue<T>(exprSomeType).has_value());
  }
};

template <typename T>
Scalar<T> CallHostRt(
    HostRuntimeWrapper func, FoldingContext &context, Scalar<T> x) {
  return GetScalarConstantValue<T>(
      func(context, {AsGenericExpr(Constant<T>{x})}))
      .value();
}

void TestHostRuntimeSubnormalFlushing() {
  using R4 = Type<TypeCategory::Real, 4>;
  if constexpr (std::is_same_v<host::HostType<R4>, float>) {
    Fortran::parser::CharBlock src;
    Fortran::parser::ContextualMessages messages{src, nullptr};
    Fortran::common::IntrinsicTypeDefaultKinds defaults;
    auto intrinsics{Fortran::evaluate::IntrinsicProcTable::Configure(defaults)};
    TargetCharacteristics flushingTargetCharacteristics;
    flushingTargetCharacteristics.set_areSubnormalsFlushedToZero(true);
    TargetCharacteristics noFlushingTargetCharacteristics;
    noFlushingTargetCharacteristics.set_areSubnormalsFlushedToZero(false);
    Fortran::common::LanguageFeatureControl languageFeatures;
    std::set<std::string> tempNames;
    FoldingContext flushingContext{messages, defaults, intrinsics,
        flushingTargetCharacteristics, languageFeatures, tempNames};
    FoldingContext noFlushingContext{messages, defaults, intrinsics,
        noFlushingTargetCharacteristics, languageFeatures, tempNames};

    DynamicType r4{R4{}.GetType()};
    // Test subnormal argument flushing
    if (auto callable{GetHostRuntimeWrapper("log", r4, {r4})}) {
      // Biggest IEEE 32bits subnormal power of two
      const Scalar<R4> x1{Scalar<R4>::Word{0x00400000}};
      Scalar<R4> y1Flushing{CallHostRt<R4>(*callable, flushingContext, x1)};
      Scalar<R4> y1NoFlushing{CallHostRt<R4>(*callable, noFlushingContext, x1)};
      // We would expect y1Flushing to be NaN, but some libc logf implementation
      // "workaround" subnormal flushing by returning a constant negative
      // results for all subnormal values (-1.03972076416015625e2_4). In case of
      // flushing, the result should still be different than -88 +/- 2%.
      TEST(y1Flushing.IsInfinite() ||
          std::abs(host::CastFortranToHost<R4>(y1Flushing) + 88.) > 2);
      TEST(!y1NoFlushing.IsInfinite() &&
          std::abs(host::CastFortranToHost<R4>(y1NoFlushing) + 88.) < 2);
    } else {
      TEST(false);
    }
  } else {
    TEST(false); // Cannot run this test on the host
  }
}

// Host folding of ERFC_SCALED at REAL(16), on hosts that support it (either
// __float128 through libquadmath or a binary128 long double). Reference
// values were computed with mpmath 1.4.1 at 300-bit precision and rounded to
// binary128 (round-to-nearest-even); inputs and expected results are given
// as raw words so that no host-dependent decimal conversion contaminates the
// comparison.
void TestErfcScaledFoldingReal16() {
  using R16 = Type<TypeCategory::Real, 16>;
  using Word = typename Scalar<R16>::Word;
  const auto real16{[](std::uint64_t hi, std::uint64_t lo) {
    return Scalar<R16>{Word{hi}.SHIFTL(64).IOR(Word{lo})};
  }};
  // True iff a and b, both finite and of the same sign, are at most maxUlps
  // representable values apart.
  const auto withinUlps{
      [](const Scalar<R16> &a, const Scalar<R16> &b, std::uint64_t maxUlps) {
        Word wa{a.RawBits()};
        Word wb{b.RawBits()};
        if (wa.CompareUnsigned(wb) == Fortran::evaluate::Ordering::Less) {
          std::swap(wa, wb);
        }
        Word diff{wa.SubtractSigned(wb).value};
        return diff.CompareUnsigned(Word{maxUlps}) !=
            Fortran::evaluate::Ordering::Greater;
      }};
  Fortran::parser::CharBlock src;
  Fortran::parser::ContextualMessages messages{src, nullptr};
  Fortran::common::IntrinsicTypeDefaultKinds defaults;
  auto intrinsics{Fortran::evaluate::IntrinsicProcTable::Configure(defaults)};
  TargetCharacteristics targetCharacteristics;
  Fortran::common::LanguageFeatureControl languageFeatures;
  std::set<std::string> tempNames;
  FoldingContext context{messages, defaults, intrinsics, targetCharacteristics,
      languageFeatures, tempNames};
  DynamicType r16{R16{}.GetType()};
  auto callable{GetHostRuntimeWrapper("erfc_scaled", r16, {r16})};
  if (!callable) {
    return; // This host cannot fold REAL(16); folding is legitimately absent.
  }
  const auto fold{[&](const Scalar<R16> &x) {
    return CallHostRt<R16>(*callable, context, x);
  }};
  // The results of the asymptotic-series branch (x >= 16) involve no
  // math-library calls and must reproduce exactly on every host. The
  // expected words are the algorithm's own results; each sits one ulp from
  // the correctly rounded value (series truncation): ...1fcf for 20.0 and
  // ...be12 for 1.0e6.
  TEST(fold(real16(0x4003400000000000, 0)) == // erfc_scaled(20.0)
      real16(0x3ff9cd9bc89b7354, 0x7fc6fb33fcba1fce));
  TEST(fold(real16(0x4012e84800000000, 0)) == // erfc_scaled(1.0e6)
      real16(0x3fea2ee5a03c7620, 0xf1a7ebc5b5aebe13));
  // Results that pass through exp/erfc are compared to the correctly
  // rounded true values with a small allowance, since math-library rounding
  // differs across implementations (libquadmath versions, long double
  // libm). Measured deviation on x86-64 with GCC 9.3 libquadmath: 0 ulps at
  // all three points.
  constexpr std::uint64_t maxUlps{16};
  TEST(withinUlps(fold(real16(0x3ffe000000000000, 0)), // erfc_scaled(0.5)
      real16(0x3ffe3b3bc3c98b0f, 0x2caaf529dbcefe16), maxUlps));
  // The true values below stay representable in binary128 well past the
  // former implementation's cutoff at -26.628, which returned HUGE from
  // there on.
  TEST(withinUlps(fold(real16(0xc003b00000000000, 0)), // erfc_scaled(-27.0)
      real16(0x441ba70cd52262b7, 0x9eae660a818fc57b), maxUlps));
  TEST(withinUlps( // erfc_scaled(-106.3)
      fold(real16(0xc005a93333333333, 0x3333333333333333)),
      real16(0x7fae0132469c65b8, 0xbc8f583f6c46a408), maxUlps));
  // Past the last representable result (x ~ -106.567), +infinity.
  Scalar<R16> atNeg107{fold(real16(0xc005ac0000000000, 0))};
  TEST(atNeg107.IsInfinite() && !atNeg107.IsNegative());
  // Specials.
  TEST(fold(real16(0x7fff800000000000, 0)).IsNotANumber()); // NaN
  Scalar<R16> atPosInf{fold(real16(0x7fff000000000000, 0))};
  TEST(atPosInf.IsZero() && !atPosInf.IsNegative()); // erfc_scaled(+Inf)->+0
  Scalar<R16> atNegInf{fold(real16(0xffff000000000000, 0))};
  TEST(atNegInf.IsInfinite() && !atNegInf.IsNegative());
  TEST(fold(real16(0x0000000000000000, 0)) == // erfc_scaled(+0) == 1
      real16(0x3fff000000000000, 0));
  TEST(fold(real16(0x8000000000000000, 0)) == // erfc_scaled(-0) == 1
      real16(0x3fff000000000000, 0));
}

int main() {
  RunOnTypes<TestGetScalarConstantValue, AllIntrinsicTypes>::Run();
  TestHostRuntimeSubnormalFlushing();
  TestErfcScaledFoldingReal16();
  return testing::Complete();
}
