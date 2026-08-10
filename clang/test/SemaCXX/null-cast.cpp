// RUN: %clang_cc1 -fsyntax-only -verify %s

struct A {};
struct B : virtual A {};

void foo() {
  (void)static_cast<A&>(*(B *)0); // expected-warning {{binding dereferenced null pointer to reference has undefined behavior}}
}
