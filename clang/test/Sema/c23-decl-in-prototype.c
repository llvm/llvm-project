// RUN: %clang_cc1 -fsyntax-only -verify -std=c23 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c23 -Wvisibility %s

// In C23 mode, we only want to diagnose a declaration in a prototype if that
// declaration is for an incomplete tag type. Otherwise, we silence the
// diagnostic because the function could be called with a compatible type.

void f(struct Incomplete); // expected-warning {{will not be visible outside of this function}}
void g(struct Complete { int x; });

struct A {
  struct B {
    int j; // #j
  } b;
};

void complicated(struct A { struct B { int j; } b; }); // Okay

void also_complicated(struct A { struct B { int glorx; } b; }); // expected-error {{type 'struct B' has incompatible definitions}} \
                                                                   expected-note {{field has name 'glorx' here}} \
                                                                   expected-note@#j {{field has name 'j' here}}
