// RUN: %clang_cc1 -verify %s

struct S; // expected-note 2{{forward declaration of 'S'}}
extern S a;
extern S f(); // expected-note {{'f' declared here}}
extern void g(S a);

void h() {
  g(a); // expected-error {{argument type 'S' is incomplete}}
  f(); // expected-error {{calling 'f' with incomplete return type 'S'}}
}
