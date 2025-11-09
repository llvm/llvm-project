// clang/test/SemaCXX/decltype-diagnostic-print.cpp
// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

template <typename T>
struct TestAssert {
  // This static_assert will fail, forcing the compiler to print the name
  // of the template instantiation in its diagnostic "note".
  static_assert(sizeof(T) == 0, "Static assert to check type printing");
};

struct MySimpleType {};
struct MyOtherType {};

void test() {
  // This will fail the static_assert.
  TestAssert<decltype(MySimpleType{})> test1;
  // expected-error@6 {{Static assert to check type printing}}
  // expected-note@16 {{in instantiation of template class 'TestAssert<MySimpleType>' requested here}}
  // CHECK-NOT: decltype(MySimpleType{})

  TestAssert<decltype(MyOtherType{})> test2;
  // expected-error@6 {{Static assert to check type printing}}
  // expected-note@22 {{in instantiation of template class 'TestAssert<MyOtherType>' requested here}}
  // CHECK-NOT: decltype(MyOtherType{})
}