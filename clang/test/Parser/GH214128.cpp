// RUN: %clang_cc1 -std=c++17 -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s

i < 0;
#if __cplusplus <= 201703L
// expected-error@-2 {{no template named 'i'}}
#endif
// expected-error@-4 {{expected '>'}}
// expected-note@-5 {{to match this '<'}}
#if __cplusplus > 201703L
// expected-warning@-7 {{declaration does not declare anything}}
#endif

void f() {
  struct __is_pod; // expected-warning {{keyword '__is_pod' will be made available as an identifier for the remainder of the translation unit}}
  struct __is_pod;
}
