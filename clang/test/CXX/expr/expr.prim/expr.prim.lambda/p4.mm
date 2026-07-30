// RUN: %clang_cc1 -fsyntax-only -std=c++11 %s -verify

@interface A
@end

void test_result_type() {
  auto l1 = [] () -> A { }; // expected-error{{interface type 'A' cannot be returned by value; did you forget * in 'A'?}}
}
