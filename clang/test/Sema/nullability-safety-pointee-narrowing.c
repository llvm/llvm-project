// A null check on *pp narrows the pointee of a pointer-to-pointer, so **pp
// is accepted on the checked path. Changing pp, storing through it, or
// changing the local it points to drops the fact.
//
// RUN: %clang_cc1 -fsyntax-only -fnullability-safety -std=c11 -verify %s
// RUN: %clang_cc1 -fsyntax-only -fnullability-safety -fnullability-default=nonnull -std=c11 -verify %s

void use(int *_Nonnull);

int truthy(int *_Nullable *_Nullable pp) {
  if (pp && *pp)
    return **pp;
  return 0;
}

int compare(int *_Nullable *_Nonnull pp) {
  if (*pp != 0)
    return **pp;
  return 0;
}

int early_return(int *_Nullable *_Nonnull pp) {
  if (!*pp)
    return 0;
  use(*pp);
  return **pp;
}

int unchecked(int *_Nullable *_Nonnull pp) {
  return **pp; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

int store_through(int *_Nullable *_Nonnull pp, int *_Nullable q) {
  if (*pp) {
    *pp = q;
    return **pp; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
  }
  return 0;
}

int reassign(int *_Nullable *_Nonnull pp, int *_Nullable *_Nonnull qq) {
  if (*pp) {
    pp = qq;
    return **pp; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
  }
  return 0;
}

int one_branch(int *_Nullable *_Nonnull pp, int c) {
  if (c && *pp) {
  }
  return **pp; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

int target_changed(int *_Nullable p) {
  int *_Nullable *_Nonnull pp = &p;
  if (*pp) {
    p = 0;
    return **pp; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
  }
  return 0;
}
