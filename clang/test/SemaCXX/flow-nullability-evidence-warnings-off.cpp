// -Rnullsafe-evidence must be independently requestable: the evidence remarks
// are produced by the whole-TU flow analysis, which is gated on enabled
// diagnostics. With every flow warning group suppressed (-Wno-flow-nullability)
// but evidence remarks requested, the analysis must still run and emit remarks.
//
// Regression test: the gate previously checked only the flow WARNING groups, so
// -Wno-flow-nullability skipped the analysis entirely and no remark fired.
//
// Evidence remarks fire; no warnings (they are all suppressed) -> -verify only
// sees the expected remarks.
//
// RUN: %clang_cc1 -fsyntax-only -fflow-sensitive-nullability -fnullability-default=nullable -Wno-nullable-to-nonnull-conversion -Wno-flow-nullability -std=c++17 -Rnullsafe-evidence %s -verify

struct Widget {
  int x;
};

// Provably returns non-null (address-of) -> all-returns-nonnull summary remark
// must fire even though flow warnings are off.
int *getX(Widget *_Nonnull w) { // expected-remark{{function 'getX' always returns a non-null pointer}}
  return &w->x; // expected-remark-re{{function 'getX' of global scope (declared at {{.*}}) returns nonnull}}
}

// A dereference of a nullable pointer would normally warn, but -Wno-flow-
// nullability suppresses it. The analysis still runs (proving the gate change);
// no warning is expected here.
int deref(int *_Nullable p) {
  return *p; // no warning: -Wno-flow-nullability
}
