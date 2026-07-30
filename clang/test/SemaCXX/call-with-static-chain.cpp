// RUN: %clang_cc1 -fsyntax-only -verify %s

int &f();

struct A {
  void f();
};

typedef int I;

void g() {
  __builtin_call_with_static_chain(f(), f) = 42;
  __builtin_call_with_static_chain(A().f(), f); // expected-error {{first argument to __builtin_call_with_static_chain must be a non-member call expression}}
  __builtin_call_with_static_chain((42).~I(), f); // expected-error {{first argument to __builtin_call_with_static_chain must not be a pseudo-destructor call}}
}
