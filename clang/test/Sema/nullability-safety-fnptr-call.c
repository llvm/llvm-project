// Calling through a _Nullable function pointer is a dereference, whether
// written f() or (*f)().
//
// RUN: %clang_cc1 -fsyntax-only -fnullability-safety -std=c11 -verify %s
// RUN: %clang_cc1 -fsyntax-only -fnullability-safety -fnullability-default=nonnull -std=c11 -verify %s
// RUN: %clang_cc1 -fsyntax-only -fnullability-safety -std=c11 -Wno-nullability-safety-dereference -verify=off %s

typedef int (*fn)(void);

struct ops {
  int (*_Nullable cb)(void);
  fn _Nonnull must;
  fn plain;
};

int direct(fn _Nullable f) {
  return f(); // expected-warning {{calling nullable function pointer 'fn _Nullable'}} expected-note {{add a null check before calling}}
}

int star(fn _Nullable f) {
  return (*f)(); // expected-warning {{dereference of nullable pointer}} expected-note {{}}
}

int member(struct ops *_Nonnull s) {
  return s->cb(); // expected-warning {{calling nullable function pointer}} expected-note {{}}
}

int member_checked(struct ops *_Nonnull s) {
  if (s->cb)
    return s->cb();
  return 0;
}

int guarded(fn _Nullable f) {
  if (!f)
    return 0;
  return f();
}

int nonnull_ok(fn _Nonnull f, struct ops *_Nonnull s) {
  return f() + s->must() + s->plain();
}

int assigned_null(fn _Nonnull g) {
  fn h = g;
  h = 0;
  return h(); // expected-warning {{calling nullable function pointer}} expected-note {{}}
}

int named_function(void);
int calls_function(void) {
  return named_function();
}

// off-no-diagnostics
