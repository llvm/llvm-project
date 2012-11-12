// RUN: %clang_cc1 -fsyntax-only -std=c++11 %s -verify

void test_nonaggregate(int i) {
  auto lambda = [i]() -> void {}; // expected-note 3{{candidate constructor}}
  decltype(lambda) foo = { 1 }; // expected-error{{no matching constructor}}
}
