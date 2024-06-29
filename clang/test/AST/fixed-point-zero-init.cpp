// RUN: %clang_cc1 -ffixed-point %s -verify
// expected-no-diagnostics

constexpr _Accum a[2] = {};
static_assert(a[0] == 0 && a[0] != 1);
