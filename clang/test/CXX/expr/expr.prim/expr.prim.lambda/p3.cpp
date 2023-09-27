// RUN: %clang_cc1 -fsyntax-only -std=c++11 %s -verify
// RUN: %clang_cc1 -fsyntax-only -std=c++14 %s -verify
// RUN: %clang_cc1 -fsyntax-only -std=c++17 %s -verify

void test_nonaggregate(int i) {
  auto lambda = [i]() -> void {}; // expected-note 2{{candidate constructor}}
  decltype(lambda) foo = { 1 }; // expected-error{{no matching constructor}}
  static_assert(__is_literal(decltype(lambda)) == (__cplusplus >= 201703L), "");

  auto lambda2 = []{}; // expected-note 2{{candidate constructor}}
  decltype(lambda2) bar = {}; // expected-error{{no matching constructor}}
  static_assert(__is_literal(decltype(lambda2)) == (__cplusplus >= 201703L), "");
}

constexpr auto literal = []{};
#if __cplusplus < 201703L
// expected-error@-2 {{constexpr variable cannot have non-literal type}}
// expected-note@-3 {{lambda closure types are non-literal types before C++17}}
// expected-error@-4 {{a lambda expression may not appear inside of a constant expression}}
#endif
