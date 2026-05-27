// RUN: %clang_cc1 -std=c++17 -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++17 -fsyntax-only -verify -fexperimental-new-constant-interpreter %s
// expected-no-diagnostics

constexpr float InfFloat = __builtin_inff();
constexpr float NegInfFloat = -__builtin_inff();

static_assert(InfFloat == __builtin_expf(InfFloat));
static_assert(0.0f == __builtin_expf(NegInfFloat));
static_assert(1.0f == __builtin_expf(0.0f));
static_assert(0x1.5bf0a8p1f == __builtin_expf(1.0f));

constexpr double InfDouble = __builtin_inf();
constexpr double NegInfDouble = -__builtin_inf();

static_assert(InfDouble == __builtin_exp(InfDouble));
static_assert(0.0 == __builtin_exp(NegInfDouble));
static_assert(1.0 == __builtin_exp(0.0));
static_assert(0x1.5bf0a8b145769p1 == __builtin_exp(1.0));
