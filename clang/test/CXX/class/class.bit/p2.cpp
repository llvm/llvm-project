// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

struct A {
private: 
  int : 0;
};

A a = { };
A a2 = { 1 }; // expected-error{{excess elements in struct initializer}}

struct B {
  const int : 0; // expected-error{{anonymous bit-field cannot have qualifiers}}
};

B b;

void testB() {
  B b2(b);
  B b3(static_cast<B&&>(b2));
  b = b;
  b = static_cast<B&&>(b);
}
