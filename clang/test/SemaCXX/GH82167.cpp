// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++23 -fsyntax-only -verify -xobjective-c++ %s
// expected-no-diagnostics

namespace t1 {
  struct array {
    char elems[2];
  };

  template <unsigned> struct Literal {
    array arr;
    constexpr Literal() : arr("") {}
  };

  template struct Literal<0>;
} // namespace t1

namespace t2 {
  struct array {
    char elems[2];
  };

  template <unsigned> struct Literal {
    array arr;
    constexpr Literal() : arr(@encode(int)) {}
  };

  template struct Literal<0>;
} // namespace t2
