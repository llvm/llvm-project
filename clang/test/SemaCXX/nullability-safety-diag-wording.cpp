// Warnings print the pointer type without the _Null_unspecified tag that
// nullable mode adds internally, keep a tag the user wrote, and name an
// unnamed nonnull parameter by position.

// RUN: %clang_cc1 -fsyntax-only -fnullability-safety -fnullability-default=nullable %s -verify

struct S { int x; };
void take(int *_Nonnull);
void take_named(int *_Nonnull p);
void take_second(int *a, int *_Nonnull);

void deref(int *p, S *s, int *_Nullable n) {
  *p = 1; // expected-warning {{dereference of nullable pointer 'int *'}} expected-note {{add a null check}}
  s->x = 1; // expected-warning {{dereference of nullable pointer 'S *'}} expected-note {{add a null check}}
  *n = 1; // expected-warning {{dereference of nullable pointer 'int * _Nullable'}} expected-note {{add a null check}}
}

void arith(int *p) {
  int *q = p + 1; // expected-warning {{pointer arithmetic on nullable pointer 'int *'}} expected-note {{add a null check}}
  (void)q;
}

void args(int *p, int *r, int *t) {
  take(p); // expected-warning {{passing nullable pointer to nonnull parameter 1}} expected-note {{add a null check}}
  take_named(r); // expected-warning {{passing nullable pointer to nonnull parameter 'p'}} expected-note {{add a null check}}
  take_second(nullptr, t); // expected-warning {{passing nullable pointer to nonnull parameter 2}} expected-note {{add a null check}}
}

void nested(int **pp) {
  **pp = 1; // expected-warning {{dereference of nullable pointer 'int **'}} expected-note {{add a null check}} expected-warning {{dereference of nullable pointer 'int *'}} expected-note {{add a null check}}
}

void keeps_const(int *const p, const int *const *q) {
  *p = 1; // expected-warning {{dereference of nullable pointer 'int *const'}} expected-note {{add a null check}}
  (void)**q; // expected-warning {{dereference of nullable pointer 'const int *const *'}} expected-note {{add a null check}} expected-warning {{dereference of nullable pointer 'const int *const'}} expected-note {{add a null check}}
}
