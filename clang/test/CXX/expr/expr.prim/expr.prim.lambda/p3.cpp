// RUN: %clang_cc1 -fsyntax-only -std=c++11 %s -verify

void test_nonaggregate(int i) {
  auto lambda = [i]() -> void {}; // expected-note 2{{candidate constructor}}
  decltype(lambda) foo = { 1 }; // expected-error{{no matching constructor}}
  static_assert(!__is_literal(decltype(lambda)), "");

  auto lambda2 = []{}; // expected-note 2{{candidate constructor}}
  decltype(lambda2) bar = {}; // expected-error{{no matching constructor}}
  static_assert(!__is_literal(decltype(lambda2)), "");
}
