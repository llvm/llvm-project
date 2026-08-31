// RUN: %clang_cc1 -fsyntax-only -verify %s

i < 0
// expected-error@-1 {{no template named 'i'}}
// expected-error@-2 {{expected '>'}}
// expected-note@-3 {{to match this '<'}}

void f() {
  struct __is_pod; // expected-warning {{keyword '__is_pod' will be made available as an identifier for the remainder of the translation unit}}
  struct __is_pod;
}
