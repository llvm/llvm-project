// RUN: %clang_cc1 -fsyntax-only -verify %s

namespace {
template <bool, typename>
void Foo() {}

template <int size>
void Foo() {
  int arr[size];
  // expected-error@-1 {{'arr' declared as an array with a negative size}}
}
}

void test_foo() {
  Foo<-1>();
  // expected-note@-1 {{in instantiation of function template specialization '(anonymous namespace)::Foo<-1>' requested here}}
}

template <bool, typename>
void Bar() {}

template <int size>
void Bar() {
  int arr[size];
  // expected-error@-1 {{'arr' declared as an array with a negative size}}
}

void test_bar() {
  Bar<-1>();
  // expected-note@-1 {{in instantiation of function template specialization 'Bar<-1>' requested here}}
}

