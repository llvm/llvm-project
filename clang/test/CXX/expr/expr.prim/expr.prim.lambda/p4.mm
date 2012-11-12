// RUN: %clang_cc1 -fsyntax-only -std=c++11 %s -verify

@interface A
@end

void test_result_type() {
  auto l1 = [] () -> A { }; // expected-error{{non-pointer Objective-C class type 'A' in lambda expression result}}
}
