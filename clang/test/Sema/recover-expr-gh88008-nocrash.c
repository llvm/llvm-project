// RUN: %clang_cc1 %s -verify -fsyntax-only -std=c90

struct S {
  int v;
};

struct T; // expected-note {{forward declaration of 'struct T'}}

void gh88008_nocrash(struct T *t) {
  struct S s = { .v = t->y }; // expected-error {{incomplete definition of type 'struct T'}}
}
