// RUN: %clang_cc1 %s -verify -fsyntax-only

struct simple { int i; };

void f(void) {
   struct simple s[1];
   s->i = 1;
}

typedef int x;
struct S {
  int x;
  x z;
};

void g(void) {
  struct S s[1];
  s->x = 1;
  s->z = 2;
}

int PR17762(struct simple c) {
  return c->i; // expected-error {{member reference type 'struct simple' is not a pointer; did you mean to use '.'?}}
}
