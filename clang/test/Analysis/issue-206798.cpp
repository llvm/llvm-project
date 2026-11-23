// RUN: %clang_analyze_cc1 -analyzer-checker=core -verify %s

struct S {
  int n;
};
struct T {
  S s;
};

// &r->s is the address of a field of a symbolic region, so it is symbol-based.
// A swapped null check (`nullptr == p`) must build the same constraint as the
// usual `p == nullptr`; otherwise the analyzer contradicts itself and reports a
// null dereference that cannot happen.
int gh206798_swapped(T *r) {
  S *p = &r->s;
  if (!p) {
  }
  if (nullptr == p)
    return 0;
  return p->n; // no-warning: p is known non-null after the check
}

int gh206798_normal(T *r) {
  S *p = &r->s;
  if (!p) {
  }
  if (p == nullptr)
    return 0;
  return p->n; // no-warning: p is known non-null after the check
}

// A genuine null dereference must still be flagged in either order.
int stillReportsRealBug_swapped(S *p) {
  if (nullptr == p)
    return p->n; // expected-warning{{Access to field 'n' results in a dereference of a null pointer (loaded from variable 'p')}}
  return 0;
}

int stillReportsRealBug_normal(S *p) {
  if (p == nullptr)
    return p->n; // expected-warning{{Access to field 'n' results in a dereference of a null pointer (loaded from variable 'p')}}
  return 0;
}
