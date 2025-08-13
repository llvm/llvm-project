// RUN: %clang_cc1 -fsyntax-only -verify %s

void f(...) {
  // FIXME: There's no disambiguation here; this is unambiguous.
  int g(int(...)); // expected-warning {{disambiguated}} expected-note {{paren}}
}

void h(int n..., int m); // expected-error {{expected ')'}} expected-note {{to match}}


namespace GH153445 {
void f(int = {}...);

struct S {
  void f(int = {}...);
  void g(int...);
};

void S::g(int = {}...) {}
}
