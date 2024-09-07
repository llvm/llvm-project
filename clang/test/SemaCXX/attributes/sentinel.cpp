// RUN: %clang_cc1 -fsyntax-only -verify %s

void f(int, ...) __attribute__((sentinel));

void g() {
  f(1, 2, __null);
}

typedef __typeof__(sizeof(int)) size_t;

struct S {
  S(int,...) __attribute__((sentinel)); // expected-note {{marked sentinel}}
  void a(int,...) __attribute__((sentinel)); // expected-note {{marked sentinel}}
  void* operator new(size_t,...) __attribute__((sentinel)); // expected-note {{marked sentinel}}
  void operator()(int,...) __attribute__((sentinel)); // expected-note {{marked sentinel}}
};

void class_test() {
  S s(1,2,3); // expected-warning {{missing sentinel in function call}}
  S* s2 = new (1,2,3) S(1, __null); // expected-warning {{missing sentinel in function call}}
  s2->a(1,2,3); // expected-warning {{missing sentinel in function call}}
  s(1,2,3); // expected-warning {{missing sentinel in function call}}
}
