// RUN: %clang_cc1 -triple arm-unknown-gnu -fsyntax-only -verify %s

void f(void) {
  struct EmptyStruct {};
  struct EmptyStruct S;
  __builtin_va_end(S); // no-crash, expected-error {{non-const lvalue reference to type '__builtin_va_list' cannot bind to a value of unrelated type 'struct EmptyStruct'}}
}