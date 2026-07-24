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

// Ternary self-guard: `cond ? cond : fallback` — the true arm is non-null
// because the condition tested it. Mirrors redis hyperloglog.c hllSparseAdd's
// `p = prev ? prev : sparse;` where both arms are non-null on their branch.
void test_ternary_self_guard(Entity* fallback) {
    Entity* _Nullable prev = getNullable();
    Entity* p = prev ? prev : fallback;
    p->x = 1; // OK - true arm guarded by cond, false arm nonnull
}

void test_ternary_self_guard_via_local() {
    Entity* _Nullable prev = getNullable();
    Entity stack;
    Entity* p = prev ? prev : &stack;
    p->x = 1; // OK - true arm guarded, false arm is address-of
}

// Explicit-comparison form of the same guard.
void test_ternary_ne_null_guard(Entity* fallback) {
    Entity* _Nullable prev = getNullable();
    Entity* p = (prev != nullptr) ? prev : fallback;
    p->x = 1; // OK
}

// Negative: a ternary whose selected arm can genuinely be null still warns.
void test_ternary_unguarded_arm_warns(int cond) {
    Entity* _Nullable maybe = getNullable();
    Entity* p = cond ? maybe : getNullable();
    p->x = 1; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

#pragma clang assume_nonnull end
