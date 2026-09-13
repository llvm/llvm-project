// RUN: %clang_cc1 -std=c++17 -fsyntax-only -verify -Wno-unknown-pragmas %s
// RUN: %clang_cc1 -std=c++17 -fsyntax-only -verify -fexperimental-new-constant-interpreter -Wno-unknown-pragmas %s
// RUN: %clang_cc1 -std=c++17 -fsyntax-only -verify -Wno-unknown-pragmas -fmath-errno %s
// RUN: %clang_cc1 -std=c++17 -fsyntax-only -verify -Wno-unknown-pragmas -fno-math-errno %s
// RUN: %clang_cc1 -std=c++17 -fsyntax-only -verify -Wno-unknown-pragmas -frounding-math %s
// RUN: %clang_cc1 -std=c++17 -fsyntax-only -verify -fexperimental-new-constant-interpreter -Wno-unknown-pragmas -frounding-math %s

constexpr float InfFloat = __builtin_inff();
constexpr float NegInfFloat = -__builtin_inff();
constexpr float qNaNFloat = __builtin_nanf("");
constexpr float DenormFloat = 0x1.0p-140f;
constexpr float NegDenormFloat = -0x1.0p-140f;

static_assert(__builtin_isnan(__builtin_expf(qNaNFloat)));
static_assert(InfFloat == __builtin_expf(InfFloat));
static_assert(0.0f == __builtin_expf(NegInfFloat));
static_assert(1.0f == __builtin_expf(0.0f));
static_assert(0x1.5bf0a8p1f == __builtin_expf(1.0f));
static_assert(1.0f == __builtin_expf(DenormFloat));
static_assert(1.0f == __builtin_expf(NegDenormFloat));

constexpr double InfDouble = __builtin_inf();
constexpr double NegInfDouble = -__builtin_inf();
constexpr double qNaNDouble = __builtin_nan("");
constexpr double DenormDouble = 0x1.0p-1050;
constexpr double NegDenormDouble = -0x1.0p-1050;

static_assert(__builtin_isnan(__builtin_exp(qNaNDouble)));
static_assert(InfDouble == __builtin_exp(InfDouble));
static_assert(0.0 == __builtin_exp(NegInfDouble));
static_assert(1.0 == __builtin_exp(0.0));
static_assert(0x1.5bf0a8b145769p1 == __builtin_exp(1.0));
static_assert(1.0 == __builtin_exp(DenormDouble));
static_assert(1.0 == __builtin_exp(NegDenormDouble));

// No constexpr for overflow.
static_assert(InfFloat == __builtin_expf(100.0f)); // expected-error {{static assertion expression is not an integral constant expression}}
// expected-note@-1 {{call to '__builtin_expf' raises a floating-point overflow exception}}
static_assert(InfDouble == __builtin_exp(1000.0)); // expected-error {{static assertion expression is not an integral constant expression}}
// expected-note@-1 {{call to '__builtin_exp' raises a floating-point overflow exception}}

// No constexpr for underflow: normal input yielding denormal output.
static_assert(0.0f == __builtin_expf(-88.0f)); // expected-error {{static assertion expression is not an integral constant expression}}
// expected-note@-1 {{call to '__builtin_expf' raises a floating-point underflow exception}}
static_assert(0.0 == __builtin_exp(-709.0)); // expected-error {{static assertion expression is not an integral constant expression}}
// expected-note@-1 {{call to '__builtin_exp' raises a floating-point underflow exception}}

// No constexpr for underflow: output rounding to zero.
static_assert(0.0f == __builtin_expf(-100.0f)); // expected-error {{static assertion expression is not an integral constant expression}}
// expected-note@-1 {{call to '__builtin_expf' raises a floating-point underflow exception}}
static_assert(0.0 == __builtin_exp(-1000.0)); // expected-error {{static assertion expression is not an integral constant expression}}
// expected-note@-1 {{call to '__builtin_exp' raises a floating-point underflow exception}}

// No constexpr for signaling NaN (invalid operation).
constexpr float sNaNFloat = __builtin_nansf("");
constexpr double sNaNDouble = __builtin_nans("");
static_assert(__builtin_isnan(__builtin_expf(sNaNFloat))); // expected-error {{static assertion expression is not an integral constant expression}}
// expected-note@-1 {{call to '__builtin_expf' raises an invalid floating-point operation exception}}
static_assert(__builtin_isnan(__builtin_exp(sNaNDouble))); // expected-error {{static assertion expression is not an integral constant expression}}
// expected-note@-1 {{call to '__builtin_exp' raises an invalid floating-point operation exception}}


void test_rounding_modes() {
  // Currently, llvm::APFloat::exp only supports NearestTiesToEven rounding mode.
  // Other rounding modes fail evaluation with note_constexpr_unsupported_rounding.
  // Once directed rounding modes are supported in APFloat::exp:
  // - Inexact evaluations without overflow/underflow (e.g., expf(1.0f)) should become
  //   valid constant expressions with the appropriately rounded value.
  // - Overflow and underflow cases should still fail constant expression evaluation, but
  //   emit note_constexpr_float_overflow or note_constexpr_float_underflow instead
  //   of note_constexpr_unsupported_rounding.
  {
    #pragma STDC FENV_ROUND FE_DOWNWARD
    // FIXME: Once FE_DOWNWARD is supported in APFloat::exp, this should fold to:
    // static_assert(0x1.5bf0a8p1f == __builtin_expf(1.0f));
    static_assert(1.0f == __builtin_expf(1.0f)); // expected-error {{static assertion expression is not an integral constant expression}}
    // expected-note@-1 {{cannot evaluate call to '__builtin_expf' in 'downward' rounding mode}}
    // FIXME: Once supported, these should fail with note_constexpr_float_overflow / underflow:
    static_assert(InfFloat == __builtin_expf(100.0f)); // expected-error {{static assertion expression is not an integral constant expression}}
    // expected-note@-1 {{cannot evaluate call to '__builtin_expf' in 'downward' rounding mode}}
    static_assert(InfDouble == __builtin_exp(1000.0)); // expected-error {{static assertion expression is not an integral constant expression}}
    // expected-note@-1 {{cannot evaluate call to '__builtin_exp' in 'downward' rounding mode}}
    static_assert(0.0f == __builtin_expf(-100.0f)); // expected-error {{static assertion expression is not an integral constant expression}}
    // expected-note@-1 {{cannot evaluate call to '__builtin_expf' in 'downward' rounding mode}}
    static_assert(0.0 == __builtin_exp(-1000.0)); // expected-error {{static assertion expression is not an integral constant expression}}
    // expected-note@-1 {{cannot evaluate call to '__builtin_exp' in 'downward' rounding mode}}
  }
  {
    #pragma STDC FENV_ROUND FE_TOWARDZERO
    // FIXME: Once FE_TOWARDZERO is supported in APFloat::exp, this should fold to:
    // static_assert(0x1.5bf0a8p1f == __builtin_expf(1.0f));
    static_assert(1.0f == __builtin_expf(1.0f)); // expected-error {{static assertion expression is not an integral constant expression}}
    // expected-note@-1 {{cannot evaluate call to '__builtin_expf' in 'towardzero' rounding mode}}
    // FIXME: Once supported, these should fail with note_constexpr_float_overflow / underflow:
    static_assert(InfFloat == __builtin_expf(100.0f)); // expected-error {{static assertion expression is not an integral constant expression}}
    // expected-note@-1 {{cannot evaluate call to '__builtin_expf' in 'towardzero' rounding mode}}
    static_assert(InfDouble == __builtin_exp(1000.0)); // expected-error {{static assertion expression is not an integral constant expression}}
    // expected-note@-1 {{cannot evaluate call to '__builtin_exp' in 'towardzero' rounding mode}}
    static_assert(0.0f == __builtin_expf(-100.0f)); // expected-error {{static assertion expression is not an integral constant expression}}
    // expected-note@-1 {{cannot evaluate call to '__builtin_expf' in 'towardzero' rounding mode}}
    static_assert(0.0 == __builtin_exp(-1000.0)); // expected-error {{static assertion expression is not an integral constant expression}}
    // expected-note@-1 {{cannot evaluate call to '__builtin_exp' in 'towardzero' rounding mode}}
  }
  {
    #pragma STDC FENV_ROUND FE_UPWARD
    // FIXME: Once FE_UPWARD is supported in APFloat::exp, this should fold to:
    // static_assert(0x1.5bf0aap1f == __builtin_expf(1.0f));
    static_assert(1.0f == __builtin_expf(1.0f)); // expected-error {{static assertion expression is not an integral constant expression}}
    // expected-note@-1 {{cannot evaluate call to '__builtin_expf' in 'upward' rounding mode}}
    // FIXME: Once supported, these should fail with note_constexpr_float_overflow / underflow:
    static_assert(InfFloat == __builtin_expf(100.0f)); // expected-error {{static assertion expression is not an integral constant expression}}
    // expected-note@-1 {{cannot evaluate call to '__builtin_expf' in 'upward' rounding mode}}
    static_assert(InfDouble == __builtin_exp(1000.0)); // expected-error {{static assertion expression is not an integral constant expression}}
    // expected-note@-1 {{cannot evaluate call to '__builtin_exp' in 'upward' rounding mode}}
    static_assert(0.0f == __builtin_expf(-100.0f)); // expected-error {{static assertion expression is not an integral constant expression}}
    // expected-note@-1 {{cannot evaluate call to '__builtin_expf' in 'upward' rounding mode}}
    static_assert(0.0 == __builtin_exp(-1000.0)); // expected-error {{static assertion expression is not an integral constant expression}}
    // expected-note@-1 {{cannot evaluate call to '__builtin_exp' in 'upward' rounding mode}}
  }
  {
    #pragma STDC FENV_ROUND FE_TONEAREST
    static_assert(1.0f == __builtin_expf(0.0f));
    static_assert(0x1.5bf0a8p1f == __builtin_expf(1.0f));
    static_assert(1.0 == __builtin_exp(0.0));
    static_assert(0x1.5bf0a8b145769p1 == __builtin_exp(1.0));
  }
}

void test_fenv_access() {
  #pragma STDC FENV_ACCESS ON
  static_assert(1.0f == __builtin_expf(0.0f));
  static_assert(0x1.5bf0a8p1f == __builtin_expf(1.0f));
  static_assert(1.0 == __builtin_exp(0.0));
  static_assert(0x1.5bf0a8b145769p1 == __builtin_exp(1.0));
  static_assert(InfFloat == __builtin_expf(100.0f)); // expected-error {{static assertion expression is not an integral constant expression}}
  // expected-note@-1 {{call to '__builtin_expf' raises a floating-point overflow exception}}
  static_assert(0.0f == __builtin_expf(-100.0f)); // expected-error {{static assertion expression is not an integral constant expression}}
  // expected-note@-1 {{call to '__builtin_expf' raises a floating-point underflow exception}}
}
