// RUN: %clang_cc1 -std=c++17 -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++17 -fsyntax-only -verify -fexperimental-new-constant-interpreter %s

constexpr float InfFloat = __builtin_inff();
constexpr float NegInfFloat = -__builtin_inff();
constexpr float qNaNFloat = __builtin_nanf("");

static_assert(__builtin_expf(qNaNFloat) != __builtin_expf(qNaNFloat));
static_assert(InfFloat == __builtin_expf(InfFloat));
static_assert(0.0f == __builtin_expf(NegInfFloat));
static_assert(1.0f == __builtin_expf(0.0f));
static_assert(0x1.5bf0a8p1f == __builtin_expf(1.0f));

// No constexpr for overflow.
static_assert(InfFloat == __builtin_expf(100.0f)); // expected-error {{static assertion expression is not an integral constant expression}}

constexpr double InfDouble = __builtin_inf();
constexpr double NegInfDouble = -__builtin_inf();
constexpr double qNaNDouble = __builtin_nan("");

static_assert(__builtin_exp(qNaNDouble) != __builtin_exp(qNaNDouble));
static_assert(InfDouble == __builtin_exp(InfDouble));
static_assert(0.0 == __builtin_exp(NegInfDouble));
static_assert(1.0 == __builtin_exp(0.0));
static_assert(0x1.5bf0a8b145769p1 == __builtin_exp(1.0));

// No constexpr for overflow.
static_assert(InfDouble == __builtin_exp(1000.0)); // expected-error {{static assertion expression is not an integral constant expression}}
