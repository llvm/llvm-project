// RUN: %clang_cc1 -fsyntax-only -pedantic -verify %s

struct C {
  static int (C::* a);
};

typedef void (C::*pmfc)();

void g(pmfc) {
  C *c;
  c->*pmfc(); // expected-error {{invalid use of pointer to member type after ->*}}
  C c1;
  c1.*pmfc(); // expected-error {{invalid use of pointer to member type after .*}}
  c->*(pmfc()); // expected-error {{invalid use of pointer to member type after ->*}}
  c1.*((pmfc())); // expected-error {{invalid use of pointer to member type after .*}}
}

int a(C* x) { 
  return x->*C::a; 
}

