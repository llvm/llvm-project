// RUN: %clang_cc1 -std=c++20 -verify %s
// expected-no-diagnostics

namespace SpaceshipImpliesEq {
  template<typename T> struct A {
    int operator<=>(const A&) const = default;
    constexpr bool f() { return operator==(*this); }
  };
  static_assert(A<int>().f());
}
