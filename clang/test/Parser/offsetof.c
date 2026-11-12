// RUN: %clang_cc1 -fsyntax-only -verify %s

struct a { struct { int b; } x[2]; };

int a = __builtin_offsetof(struct a, x; // expected-error{{expected ')'}} expected-note{{to match this '('}}
// FIXME: This actually shouldn't give an error
int b = __builtin_offsetof(struct a, x->b); // expected-error{{expected ')'}} expected-note{{to match this '('}}
