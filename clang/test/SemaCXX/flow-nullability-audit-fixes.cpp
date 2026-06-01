// flow-nullability-audit-fixes.cpp - Tests for audit-identified bugs.
//
// Each section pins behavior for a bug found during audit. These tests
// encode the EXPECTED (post-fix) behavior: narrowing must drop when a
// store can invalidate it, and warnings must NOT fire on code that is
// actually safe (attribute-annotated parameters).
//
// Leave -Wnullable-to-nonnull-conversion enabled so we can observe the
// flow-sensitive nullable-assignment warning ("assigning nullable pointer ...").
// RUN: %clang_cc1 -fsyntax-only -fflow-sensitive-nullability -fnullability-default=nullable -std=c++20 %s -verify

#pragma clang assume_nonnull begin

struct Node { int value; };

// ===----------------------------------------------------------------------===//
// Bug #2: ternary initializer loses nullability when one arm is nonnull
// and the other nullable, because isNullableInit() doesn't look inside
// ConditionalOperator. The common type loses the qualifier and the init
// slips through.
// ===----------------------------------------------------------------------===//

void ternary_nullptr_arm(bool cond, Node * other) {
    // One arm is nonnull (other), one arm is a null pointer constant.
    // The common type is `Node *` without nullability qualifier — the
    // type-only check misses it. Must warn because the RHS can be null.
    Node * _Nonnull p = cond ? other : (Node*)0; // expected-warning{{assigning nullable pointer to nonnull variable}} expected-note{{add a null check before assigning}}
    (void)p;
}

// ===----------------------------------------------------------------------===//
// Bug #3: smart-ptr operator= on this-member not invalidated
// `this->sp = nullptr` must clear NarrowedThisMembers so a subsequent
// deref warns.
// ===----------------------------------------------------------------------===//

namespace std {
template <typename T> struct unique_ptr {
    T *ptr;
    unique_ptr();
    unique_ptr(decltype(nullptr));
    T *operator->() const;
    T &operator*() const;
    unique_ptr &operator=(decltype(nullptr));
    explicit operator bool() const;
};
} // namespace std

struct HasSmart {
    std::unique_ptr<Node> sp;
    void bug3_this_member_reset_via_assign() {
        if (sp) {
            this->sp = nullptr;       // must invalidate narrowing
            (void)this->sp->value;    // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
        }
    }
};

// ===----------------------------------------------------------------------===//
// Bug #1/#4: store-through-alias `*pp = nullptr` does not invalidate the
// underlying local. If the analysis tracks that pp aliases &p, then a
// write through *pp must erase p's narrowing.
// ===----------------------------------------------------------------------===//

void bug1_store_through_alias(Node * _Nullable p) {
    if (p) {
        // p is narrowed here by the null check.
        Node ** pp = &p;   // pp aliases &p
        *pp = nullptr;     // store through pp kills p's narrowing
        (void)p->value;    // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
    }
}

// ===----------------------------------------------------------------------===//
#pragma clang assume_nonnull end

// ===----------------------------------------------------------------------===//
// FP #1 (S4): __attribute__((nonnull)) parameters must be narrowed on
// entry so derefs inside the body don't warn. Exercised OUTSIDE
// assume_nonnull, where the default nullability is "nullable" — the
// attribute is the only thing declaring the param nonnull.
// ===----------------------------------------------------------------------===//

__attribute__((nonnull(1)))
void fp_attribute_nonnull_single(Node * p) {
    // Must NOT warn — caller contract guarantees p is non-null.
    (void)p->value;
}

__attribute__((nonnull))
void fp_attribute_nonnull_all(Node * p, Node * q) {
    // Must NOT warn on either.
    (void)p->value;
    (void)q->value;
}

// expected-no-diagnostics-in-this-function-is-implicit
// (verify mode only checks marked expectations; unmarked absence is the test)
