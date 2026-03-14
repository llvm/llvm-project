// RUN: %clang_cc1 -pedantic-errors -std=c++2a -emit-pch  %s -o %t
// RUN: %clang_cc1 -pedantic-errors -std=c++2a -include-pch %t -verify %s
// RUN: %clang_cc1 -pedantic-errors -std=c++2a -include-pch %t -emit-llvm %s -o -


#ifndef HEADER
#define HEADER

#include "Inputs/std-compare.h"
constexpr auto foo() {
  return (42 <=> 101);
}

inline auto bar(int x) {
  return (1 <=> x);
}

struct X {
  int a;
  friend constexpr std::strong_ordering operator<=>(const X &x, const X &y) {
    return x.a <=> y.a;
  }
};
constexpr auto baz(int x) {
  return X{3} < X{x};
}

#else

// expected-no-diagnostics

static_assert(foo() < 0);

auto bar2(int x) {
  return bar(x);
}

static_assert(!baz(3));
static_assert(baz(4));

#endif
