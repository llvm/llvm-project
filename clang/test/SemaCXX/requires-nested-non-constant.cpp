// RUN: %clang_cc1 -std=c++23 -fsyntax-only -verify %s

template <class C> class A {
  void f() {
    auto result = []() constexpr {
      return requires (int x) {
        requires (x > 0) && (x < 10); // expected-error {{nested requirement is not a constant expression}}
      };
    }();
  }
};
