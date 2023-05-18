// RUN: %clang_cc1 -std=c++20 -verify %s
// expected-no-diagnostics

namespace FunctionTemplate {
  template<typename> struct S {
    template<int> auto foo();

    // Check that we don't confuse the depth-1 level-0 parameter of the generic
    // lambda with the depth-1 level-0 parameter of the primary 'foo' template.
    template<> constexpr auto foo<1>() {
      return [](auto x) { return x; };
    }
  };

  static_assert(S<void>().template foo<1>()(2) == 2);
}
