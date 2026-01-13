// RUN: %clang_cc1 -std=c++23 -fsyntax-only -verify %s

template <class C> class A {
  void f() {
    auto result = []() constexpr {
      return requires (int x) { // expected-note {{declared here}}
        requires (x > 0) && (x < 10); // expected-error {{substitution into constraint expression resulted in a non-constant expression}} \
                                       // expected-note {{while checking the satisfaction of nested requirement requested here}} \
                                       // expected-note {{function parameter 'x' with unknown value cannot be used in a constant expression}}
      };
    }();
  }
};
