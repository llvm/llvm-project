// RUN: %clang_cc1 -fsyntax-only %s -verify
// PR5692

enum x;            // expected-note   {{forward declaration}}
extern struct y a; // expected-note   {{forward declaration}}
extern union z b;  // expected-note 2 {{forward declaration}}

void foo() {
  (enum x)1;   // expected-error {{cast to incomplete type}}
  (struct y)a; // expected-error {{cast to incomplete type}}
  (union z)b;  // expected-error {{cast to incomplete type}}
  (union z)1;  // expected-error {{cast to incomplete type}}
}

