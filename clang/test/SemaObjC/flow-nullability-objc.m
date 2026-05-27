// Smoke test: flow-sensitive nullability in Objective-C.
// Verifies that the analysis works with ObjC pointer types and
// _Nullable/_Nonnull annotations that originated in the ObjC ecosystem.
//
// RUN: %clang_cc1 -fsyntax-only -fflow-sensitive-nullability -fnullability-default=nullable -Wno-unused-value %s -verify

void deref_nullable_raw(int * _Nullable p) {
    *p; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
    if (p) {
        *p; // OK — narrowed
    }
}

void nonnull_param(int * _Nonnull safe) {
    *safe; // OK — _Nonnull
}

void narrowing(int * _Nullable a, int * _Nullable b) {
    if (a != 0) {
        *a; // OK — narrowed
    }
    if (b) {
        *b; // OK — narrowed
    }
    *a; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}
