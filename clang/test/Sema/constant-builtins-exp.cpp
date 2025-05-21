// RUN: %clang_cc1 -std=c++17 -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++17 -fsyntax-only -verify -fexperimental-new-constant-interpreter %s
// REQUIRES: llvm_integrate_libc
// expected-no-diagnostics

constexpr float Inf = __builtin_inff();
constexpr float NegInf = -__builtin_inff();

static_assert(Inf == __builtin_expf(Inf));
static_assert(0.0f == __builtin_expf(NegInf));
static_assert(1.0f == __builtin_expf(0.0f));
static_assert(0x1.5bf0a8p1f == __builtin_expf(1.0f));
