// Under -fnullability-default=nonnull an unannotated pointer parameter of a
// function declared outside system headers means _Nonnull at the call site,
// matching how the callee's body reads it. System-header declarations stay
// unspecified. In the default mode unannotated parameters are not checked.
//
// RUN: %clang_cc1 -fsyntax-only -fnullability-safety -fnullability-default=nonnull -std=c11 -verify=nn %s
// RUN: %clang_cc1 -fsyntax-only -fnullability-safety -std=c11 -verify=def %s

# 1 "sys.h" 1 3
void sys_take(int *p);
# 12 "nullability-safety-default-nonnull-params.c" 2

struct L { int x; };
struct L *_Nullable split(void);
void assign(struct L *l);
void assign_nullable(struct L *_Nullable l);
void assign_unspecified(struct L *_Null_unspecified l);
void assign_second(int n, struct L *l);

void a(void) {
  assign(split()); // nn-warning {{passing nullable pointer to nonnull parameter 'l'}} nn-note {{}}
}
void c(void) {
  struct L *p = split();
  assign(p); // nn-warning {{passing nullable pointer to nonnull parameter 'l'}} nn-note {{}}
}
void null_literal(void) {
  assign(0); // nn-warning {{passing nullable pointer to nonnull parameter 'l'}} nn-note {{}}
  assign_second(1, 0); // nn-warning {{passing nullable pointer to nonnull parameter 'l'}} nn-note {{}}
}

// Explicit annotations on the parameter win over the default.
void explicit_param(void) {
  assign_nullable(split());
  assign_unspecified(split());
  assign_nullable(0);
}

// Checked values are fine.
void checked(void) {
  struct L *p = split();
  if (p)
    assign(p);
}

// System-header declarations stay unspecified.
void system_param(int *_Nullable q) {
  sys_take(q);
  sys_take(0);
}

// Function pointers and variadic arguments are not checked.
void variadic(int n, ...);
void indirect(void (*_Nonnull fp)(struct L *)) {
  fp(split());
  variadic(1, split());
}

// def-no-diagnostics

// A callee whose definition tests the parameter against null accepts null,
// so its callers are not held to non-null.
void tolerant(struct L *l) { if (l) l->x = 1; }
void tolerant_eq(struct L *l) { if (l == 0) return; l->x = 1; }
void calls_tolerant(void) {
  tolerant(split());
  tolerant(0);
  tolerant_eq(0);
}
