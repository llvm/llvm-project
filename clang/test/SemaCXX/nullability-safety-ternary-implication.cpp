// RUN: %clang_cc1 -fsyntax-only -fnullability-safety -fnullability-default=nonnull -std=c++17 %s -verify

struct R { int x; R *next(); };
struct H { R *renderer(); };

int pointer_antecedent_reverse(H *hin, bool on) {
  H *h = on ? hin : nullptr;
  R *r = h ? h->renderer() : nullptr;
  if (r != nullptr)
    return h->renderer()->x + r->x;
  return 0;
}

int chained(R *a0, bool on) {
  R *a = on ? a0 : nullptr;
  R *b = a ? a->next() : nullptr;
  R *c = b ? b->next() : nullptr;
  if (c != nullptr && a != nullptr)
    return a->x + b->x;
  return 0;
}

int comparison_antecedent(R *p0, bool on) {
  R *p = on ? p0 : nullptr;
  R *q = p != nullptr ? p->next() : nullptr;
  if (q)
    return p->x;
  return 0;
}

int conjunction_antecedent(R *p0, bool on, bool flag) {
  R *p = on ? p0 : nullptr;
  R *q = (p != nullptr && flag) ? p->next() : nullptr;
  if (q)
    return p->x;
  return 0;
}

int antecedent_reassigned(R *p0, bool on) {
  R *p = on ? p0 : nullptr;
  R *q = p ? p->next() : nullptr;
  p = nullptr;
  if (q)
    return p->x; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
  return 0;
}

int implied_by_reassigned(R *p0, R *other, bool on) {
  R *p = on ? p0 : nullptr;
  R *q = p ? p->next() : nullptr;
  q = other;
  if (q)
    return p->x; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
  return 0;
}

int antecedent_reassigned_through_alias(R *p0, bool on) {
  R *_Nullable p = on ? p0 : nullptr;
  R *_Nullable *pp = &p;
  R *q = p ? p->next() : nullptr;
  *pp = nullptr;
  if (q)
    return p->x; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
  return 0;
}

int no_implication_without_null_arm(R *p0, R *other, bool on) {
  R *p = on ? p0 : nullptr;
  R *q = p ? p->next() : other;
  if (q)
    return p->x; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
  return 0;
}

int null_true_arm(R *p0, bool on) {
  R *p = on ? p0 : nullptr;
  R *q = p == nullptr ? nullptr : p->next();
  if (q)
    return p->x;
  return 0;
}

int null_true_arm_disjunction(R *p0, bool on, bool flag) {
  R *p = on ? p0 : nullptr;
  R *q = (p == nullptr || flag) ? nullptr : p->next();
  if (q)
    return p->x;
  return 0;
}

int disjunction_implies_nothing(R *p0, bool on, bool flag) {
  R *p = on ? p0 : nullptr;
  R *q = (p != nullptr || flag) ? p0 : nullptr;
  if (q)
    return p->x; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
  return 0;
}

int self_reference(R *p0, bool on) {
  R *p = on ? p0 : nullptr;
  R *old = p;
  p = p ? p->next() : nullptr;
  if (p)
    return old->x; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
  return 0;
}

int mutated_in_arm(R *p0, R *other, bool on) {
  R *p = on ? p0 : nullptr;
  R *q = p ? (p = nullptr, other) : nullptr;
  if (q)
    return p->x; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
  return 0;
}

int through_bool_guard(R *p0, bool on) {
  R *p = on ? p0 : nullptr;
  bool ok = p != nullptr;
  R *q = ok ? p0 : nullptr;
  if (q)
    return p->x;
  return 0;
}

int cycle(R *a0, R *b0, bool on) {
  R *a = on ? a0 : nullptr;
  R *b = a ? b0 : nullptr;
  a = b ? a : nullptr;
  if (a)
    return b->x;
  return 0;
}

int lost_at_join(R *p0, R *other, bool on, bool c) {
  R *p = on ? p0 : nullptr;
  R *q;
  if (c)
    q = p ? p->next() : nullptr;
  else
    q = other;
  if (q)
    return p->x; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
  return 0;
}

int loop_reassigned(R *p0, R *other, bool on, int n) {
  R *p = on ? p0 : nullptr;
  int sum = 0;
  for (int i = 0; i < n; ++i) {
    R *q = p ? p->next() : nullptr;
    p = nullptr;
    if (q)
      sum += p->x; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
    p = other;
  }
  return sum;
}
