// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++17 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++20 %s

struct foo {
  static constexpr bool bar() {
      return true;
  }

  template<bool B = bar()>
  static constexpr bool baz() {
      return B;
  }
};
static_assert(foo::baz(), "");

// expected-no-diagnostics
