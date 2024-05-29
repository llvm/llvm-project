// RUN: %clang_cc1 -std=c++20 %s -verify
// expected-no-diagnostics

#include "Inputs/std-compare.h"

struct S {
  _BitInt(12) a;

  constexpr operator _BitInt(12)() const { return a; }
};

// None of these used to compile because we weren't adding _BitInt types to the
// overload set for builtin operators. See GH82998.
static_assert(S{10} < 11);
static_assert(S{10} <= 11);
static_assert(S{12} > 11);
static_assert(S{12} >= 11);
static_assert(S{10} == 10);
static_assert((S{10} <=> 10) == 0);
static_assert(S{10} != 11);
static_assert(S{10} + 0 == 10);
static_assert(S{10} - 0 == 10);
static_assert(S{10} * 1 == 10);
static_assert(S{10} / 1 == 10);
static_assert(S{10} % 1 == 0);
static_assert(S{10} << 0 == 10);
static_assert(S{10} >> 0 == 10);
static_assert((S{10} | 0) == 10);
static_assert((S{10} & 10) == 10);
static_assert((S{10} ^ 0) == 10);
static_assert(-S{10} == -10);
static_assert(+S{10} == +10);
static_assert(~S{10} == ~10);

struct A {
  _BitInt(12) a;

  bool operator==(const A&) const = default;
  bool operator!=(const A&) const = default;
  std::strong_ordering operator<=>(const A&) const = default;
};

