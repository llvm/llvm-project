// RUN: %clang_cc1 -fsyntax-only -verify %s

struct A {
  void operator()(int); // expected-note {{member found by ambiguous name lookup}}
  void f(int); // expected-note {{member found by ambiguous name lookup}}
};
struct B {
  void operator()(); // expected-note {{member found by ambiguous name lookup}}
  void f() {} // expected-note {{member found by ambiguous name lookup}}
};

struct C : A, B {};

int f() {
    C c;
    c(); // expected-error {{member 'operator()' found in multiple base classes of different types}}
    c.f(10); //expected-error {{member 'f' found in multiple base classes of different types}}
    return 0;
}
