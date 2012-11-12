// RUN: %clang_cc1 -fsyntax-only -Wc++11-compat -verify %s

class A {
  friend static class B; // expected-error {{'static' is invalid in friend declarations}}
  friend extern class C; // expected-error {{'extern' is invalid in friend declarations}}
  friend auto class D; // expected-warning {{incompatible with C++11}} expected-error {{'auto' is invalid in friend declarations}}
  friend register class E; // expected-error {{'register' is invalid in friend declarations}}
  friend mutable class F; // expected-error {{'mutable' is invalid in friend declarations}}
  friend typedef class G; // expected-error {{'typedef' is invalid in friend declarations}}
};
