// RUN: %clang_cc1 -fsyntax-only -verify %s

void fun(
    void a, // expected-error{{'void' must be the first and only parameter if specified}}
    double b,
    int c,
    void d, // expected-error{{'void' must be the first and only parameter if specified}}
    int e,
    void f) // expected-error{{'void' must be the first and only parameter if specified}}
{}

void foo(
    int a,
    void, // expected-error{{'void' must be the first and only parameter if specified}}
    int b);

void bar(
    void, // expected-error{{'void' must be the first and only parameter if specified}}
    ...);

struct S {
  S(
      void,  // expected-error{{'void' must be the first and only parameter if specified}}
      void); // expected-error{{'void' must be the first and only parameter if specified}}
};
