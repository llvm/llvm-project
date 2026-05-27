// RUN: %clang_cc1 -fsyntax-only -fflow-sensitive-nullability -fnullability-default=nonnull -std=c++17 %s -verify

struct Entity {
    int x;
};

Entity* _Nullable getNullable();
Entity* getUnannotated();

#pragma clang assume_nonnull begin

void test_unannotated_param_no_warn(Entity* p) {
    p->x = 1; // OK - parameter gets _Nonnull from assume_nonnull pragma
}

void test_unannotated_star(Entity* p) {
    (*p).x = 1; // OK - parameter gets _Nonnull from pragma
}

void test_explicit_nullable_warns(Entity* _Nullable p) {
    p->x = 1; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

void test_explicit_nullable_after_check(Entity* _Nullable p) {
    if (p) {
        p->x = 1; // OK - narrowed
    }
}

void test_return_nullable_warns() {
    Entity* e = getNullable();
    e->x = 1; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

// With -fnullability-default=nonnull, unannotated pointers are treated as
// nonnull. getUnannotated() has no _Nullable, so it's safe.
void test_return_unannotated_ok() {
    Entity* e = getUnannotated();
    e->x = 1; // OK - unannotated return treated as nonnull per default
}

void test_local_nonnull_ok() {
    Entity stack;
    Entity* _Nonnull p = &stack;
    p->x = 1; // OK - explicit _Nonnull
}

#pragma clang assume_nonnull end
