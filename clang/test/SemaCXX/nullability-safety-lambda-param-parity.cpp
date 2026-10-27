// RUN: %clang_cc1 -fsyntax-only -fnullability-safety -fnullability-default=nonnull -std=c++17 %s -verify

struct V { int x; };

int unannotated_param_null_checked(bool b) {
  auto f = [](V *p) { return p ? p->x : 0; };
  V *q = nullptr;
  if (b)
    q = new V;
  return f(q);
}

int explicit_nonnull_param(bool b) {
  auto f = [](V *_Nonnull p) { return p->x; };
  V *q = nullptr;
  if (b)
    q = new V;
  return f(q); // expected-warning {{passing nullable pointer to nonnull parameter}} expected-note {{add a null check before the call}}
}

int attr_nonnull_param(bool b) {
  auto f = [](V *p) __attribute__((nonnull(2))) { return p->x; };
  V *q = nullptr;
  if (b)
    q = new V;
  return f(q); // expected-warning {{passing nullable pointer to nonnull parameter}} expected-note {{add a null check before the call}}
}

void sink(V *p) { if (p) p->x = 1; }
void function_parity(bool b) {
  V *q = nullptr;
  if (b)
    q = new V;
  sink(q);
}
