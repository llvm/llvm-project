// RUN: %clang_cc1 -triple thumbv7--windows-msvc -std=c++11 -verify -fsyntax-only %s
// expected-no-diagnostics

#include <stdarg.h>

template <typename lhs_, typename rhs_>
struct is_same { enum { value = 0 }; };

template <typename type_>
struct is_same<type_, type_> { enum { value = 1 }; };

void check() {
  va_list va;
  char *cp;
  static_assert(is_same<decltype(va), decltype(cp)>::value,
                "type mismatch for va_list");
}
