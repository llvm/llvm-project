// Flag-off no-op coverage for C. Without -fnullability-safety and
// without -fnullability-default, representative nullability-safety test cases
// must compile silently — the feature is off by default.
//
// RUN: %clang_cc1 -fsyntax-only -std=c11 -verify %s

// expected-no-diagnostics

#define NULL ((void *)0)

struct node {
  int value;
  struct node *_Nullable next;
};

struct node *_Nullable get_node(void);

int deref_arrow(struct node *_Nullable p) { return p->value; }
int deref_chain(struct node *_Nullable p) { return p->next->value; }
int deref_star(int *_Nullable p) { return *p; }
int deref_subscript(int *_Nullable p) { return p[3]; }
int deref_call_result(void) { return get_node()->value; }

int *_Nonnull null_init_nonnull(void) {
  int *_Nonnull q = NULL; // flow-on: warn_nullability_safety_null_init; off: silent
  return q;
}

#pragma clang assume_nonnull begin
int deref_in_pragma_region(int *_Nullable p) { return *p; }
#pragma clang assume_nonnull end
