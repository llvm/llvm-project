// flow-nullability-analysis.cpp - Core flow-sensitive nullability analysis tests.
//
// Consolidated from ~30 individual test files. Tests the CFG-based forward
// dataflow analysis: narrowing, dereference checking, condition decomposition,
// alias tracking, and control flow patterns.
//
// RUN: %clang_cc1 -fsyntax-only -fflow-sensitive-nullability -fnullability-default=nullable -Wno-nullable-to-nonnull-conversion -std=c++11 -fcxx-exceptions -Wno-unused-value %s -verify
// RUN: %clang_cc1 -fsyntax-only -fflow-sensitive-nullability -fnullability-default=nullable -Wno-nullable-to-nonnull-conversion -std=c++17 -fcxx-exceptions -Wno-unused-value %s -verify
// RUN: %clang_cc1 -fsyntax-only -fflow-sensitive-nullability -fnullability-default=nullable -Wno-nullable-to-nonnull-conversion -std=c++20 -fcxx-exceptions -Wno-unused-value %s -verify

// ===----------------------------------------------------------------------===//
// Common types and helpers
// ===----------------------------------------------------------------------===//

struct Entity {
    int x;
    int value() const { return x; }
};

struct Node {
    int value;
    Node * _Nullable next;
    Node * _Nullable left;
    Node * _Nullable right;
    Node * _Nullable parent;
    Node * _Nullable child;
};

struct Container {
    Node * _Nullable root;
    Node * _Nullable head;
    int size;
};

typedef unsigned long size_t;
typedef unsigned char uint8_t;

Entity * _Nullable getNullableEntity();
Entity * _Nonnull getNonnullEntity();
Node * _Nullable getNode();
Container * _Nullable getContainer();
int getInt();
int * _Nullable getNullableInt();

[[noreturn]] void fatal(const char* msg);
[[noreturn]] void abort_handler(const char* msg);
void log_msg(const char* msg);

// ===----------------------------------------------------------------------===//
// Address-of expressions
// ===----------------------------------------------------------------------===//

#pragma clang assume_nonnull begin

void addr_test_local() {
    int x = 0;
    int *p = &x;
    *p = 1; // OK - &x is nonnull
}

void addr_test_direct() {
    Entity e;
    Entity *p = &e;
    p->x = 1; // OK - &e is nonnull
}

void addr_test_member(Entity *_Nonnull obj) {
    int *p = &(obj->x);
    *p = 1; // OK - &(obj->x) is nonnull
}

void addr_test_reassign_nullable_warns() {
    int x = 0;
    int *p = &x;
    *p = 1; // OK - initially nonnull
    p = getNullableInt();
    *p = 2; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

void addr_test_nullable_control() {
    Entity *e = getNullableEntity();
    e->x = 1; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

#pragma clang assume_nonnull end

// ===----------------------------------------------------------------------===//
// Alias tracking
// ===----------------------------------------------------------------------===//

// When y = x, checking y for null also narrows x (and vice versa).

void alias_test_check_alias_narrow_original(int *_Nullable x) {
  int *y = x;
  if (y) {
    (void)*x; // no warning -- y aliases x, y is checked
  }
  (void)*x; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

void alias_test_check_original_narrow_alias(int *_Nullable x) {
  int *y = x;
  if (x) {
    (void)*y; // no warning -- x is checked, y aliases x
  }
  (void)*y; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

void alias_test_invalidated_by_reassignment(int *_Nullable x, int *_Nullable q) {
  int *y = x;
  y = q;       // y no longer aliases x
  if (y) {
    (void)*x; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
  }
}

void alias_test_invalidated_by_source_reassignment(int *_Nullable x, int *_Nullable q) {
  int *y = x;
  x = q;       // x reassigned -- alias y -> x is stale
  if (y) {
    (void)*x; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
  }
}

void alias_test_multiple(int *_Nullable x) {
  int *y = x;
  int *z = x;
  if (z) {
    (void)*x; // no warning -- z aliases x, z is checked
    (void)*y; // no warning -- y also aliases x, x is narrowed
  }
}

void alias_test_chain(int *_Nullable x) {
  int *y = x;
  int *z = y;  // z -> canonical(y) -> x
  if (z) {
    (void)*x; // no warning -- z ultimately aliases x
    (void)*y; // no warning -- y aliases x too
  }
}

void alias_test_invalidated_by_increment(int *_Nullable x) {
  int *y = x;
  x++;         // x changed -- alias is stale // expected-warning{{pointer arithmetic on nullable pointer}} expected-note{{add a null check before performing arithmetic}}
  if (y) {
    (void)*x; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
  }
}

void alias_test_early_return(int *_Nullable x) {
  int *y = x;
  if (!y)
    return;
  (void)*x; // no warning -- early return means y (and thus x) is non-null
}

int *_Nullable alias_get_ptr();

void alias_test_no_alias_for_call_result(int *_Nullable x) {
  int *y = alias_get_ptr(); // y does NOT alias x
  if (y) {
    (void)*x; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
  }
}

void alias_test_and_shortcircuit(int *_Nullable a, int *_Nullable b) {
  int *x = a;
  int *y = b;
  if (x && y) {
    (void)*a; // no warning -- x aliases a, x is checked
    (void)*b; // no warning -- y aliases b, y is checked
  }
}

void alias_test_negated_and(int *_Nullable a, int *_Nullable b) {
  int *x = a;
  int *y = b;
  if (!(x && y))
    return;
  (void)*a; // no warning
  (void)*b; // no warning
}

void alias_test_with_bool_guard(int *_Nullable x) {
  int *y = x;
  bool ok = (y != nullptr);
  if (ok) {
    (void)*x; // no warning -- bool guard resolves y, alias propagates to x
  }
}

// ===----------------------------------------------------------------------===//
// AND short-circuit narrowing
// ===----------------------------------------------------------------------===//

#pragma clang assume_nonnull begin

void and_test_basic(Node* _Nullable p) {
    if (p && p->value == 42) {
        p->value = 0; // OK - p narrowed by && LHS
    }
}

void and_test_star(Node* _Nullable p) {
    if (p && (*p).value == 42) {
        (*p).value = 0; // OK
    }
}

void and_test_chained_no_warning(Node* _Nullable p) {
    if (p && p->next && p->next->value > 0) {
        p->next->value = 0; // OK - member narrowing works throughout
    }
}

void and_test_two_vars(Node* _Nullable p, Node* _Nullable q) {
    if (p && q) {
        p->value = q->value; // OK - both narrowed
    }
}

void and_test_three_vars(Node* _Nullable a, Node* _Nullable b, Node* _Nullable c) {
    if (a && b && c) {
        a->value = b->value + c->value; // OK
    }
}

void and_test_member_two_part(Node* _Nullable p) {
    if (p && p->next) {
        p->next->value = 1; // OK - both p and p->next narrowed
    }
}

#pragma clang assume_nonnull end

// ===----------------------------------------------------------------------===//
// Array subscript
// ===----------------------------------------------------------------------===//

#pragma clang assume_nonnull begin

void subscript_test_warns(int* _Nullable p) {
    p[0] = 42; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

void subscript_test_after_check(int* _Nullable p) {
    if (p) {
        p[0] = 42; // OK - narrowed by check
    }
}

void subscript_test_offset(int* _Nullable p) {
    p[5] = 42; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

void subscript_test_fixed_array_no_warn() {
    int arr[4] = {1, 2, 3, 4};
    arr[0] = 10; // OK - fixed-size array, not a pointer
}

struct SubscriptS {
    float gridColor[4];
    struct { int x; } nested[2];
};

void subscript_test_member_fixed_array(SubscriptS s) {
    float r = s.gridColor[0]; // OK - fixed-size array member
    int x = s.nested[1].x;   // OK - fixed-size array member
}

#pragma clang assume_nonnull end

// ===----------------------------------------------------------------------===//
// Arrow and star dereference
// ===----------------------------------------------------------------------===//

Entity* _Nullable getHead();
Entity* _Nullable getChest();

#pragma clang assume_nonnull begin

void arrow_test_warns(Entity* _Nullable p) {
    p->x = 1;              // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
    int v = p->value();     // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

void arrow_test_after_null_check(Entity* _Nullable p) {
    if (p) {
        p->x = 1;          // OK - narrowed to nonnull
        int v = p->value(); // OK - narrowed to nonnull
    }
}

void arrow_test_no_check() {
    Entity* head = getHead();
    head->x = 1;            // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

void arrow_test_with_check() {
    Entity* head = getHead();
    if (!head) return;
    head->x = 1;            // OK - narrowed to nonnull
}

void arrow_test_star_still_works(Entity* _Nullable p) {
    (*p).x = 1;             // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

void arrow_test_star_after_check(Entity* _Nullable p) {
    if (p) {
        (*p).x = 1;         // OK - narrowed to nonnull
    }
}

// Member field assignment invalidation.
struct ArrowContainer {
    Entity* _Nullable child;

    void test_member_assign_invalidates() {
        if (child) {
            child->x = 1;    // OK -- narrowed
            child = nullptr;
            child->x = 1;    // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
        }
    }
};

// Member field assignment re-narrowing from _Nonnull source.
Entity* _Nonnull get_nonnull_entity();

struct MemberAssignNarrow {
    Entity* _Nullable field;

    // Assigning from _Nonnull return value narrows the member.
    void test_nonnull_return(Entity* _Nonnull safe) {
        field = get_nonnull_entity();
        field->x = 1; // OK -- narrowed by _Nonnull assignment
    }

    // Assigning from _Nonnull parameter narrows the member.
    void test_nonnull_param(Entity* _Nonnull safe) {
        field = safe;
        field->x = 1; // OK -- narrowed by _Nonnull parameter
    }

    // Assigning address-of narrows the member.
    void test_address_of() {
        Entity local;
        field = &local;
        field->x = 1; // OK -- narrowed by address-of
    }

    // Assigning from new narrows the member.
    void test_new_expr() {
        field = new Entity;
        field->x = 1; // OK -- narrowed by new
    }

    // Assigning from nullable does NOT narrow.
    void test_nullable_assign(Entity* _Nullable maybe) {
        field = maybe;
        field->x = 1; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
    }

    // Re-narrowing after invalidation.
    void test_renarrow_after_null() {
        field = nullptr;
        field->x = 1; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
        field = get_nonnull_entity();
        field->x = 1; // OK -- re-narrowed
    }
};

// Non-this member assignment narrowing (var->field).
Node* _Nonnull get_nonnull_node();

void test_var_member_nonnull_assign(Node* _Nullable container) {
    if (!container) return;
    container->next = get_nonnull_node();
    container->next->value = 1; // OK -- narrowed by _Nonnull assignment
}

void test_var_member_nullable_assign(Node* _Nullable container, Node* _Nullable maybe) {
    if (!container) return;
    container->next = maybe;
    container->next->value = 1; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

#pragma clang assume_nonnull end

// ===----------------------------------------------------------------------===//
// Assignment in condition
// ===----------------------------------------------------------------------===//

#pragma clang assume_nonnull begin

Node *_Nullable assign_get_next(Node *n);

void assign_cond_while_ne_null(Node *_Nullable head) {
    Node *p;
    while ((p = assign_get_next(head)) != nullptr) { // expected-warning{{passing nullable pointer to nonnull parameter}} expected-note{{add a null check before the call}}
        p->value = 1; // OK -- p narrowed by != nullptr
    }
}

void assign_cond_while_truthiness(Node *_Nullable head) {
    Node *p;
    while ((p = assign_get_next(head))) { // expected-warning{{passing nullable pointer to nonnull parameter}} expected-note{{add a null check before the call}}
        p->value = 1; // OK -- p narrowed by truthiness
    }
}

void assign_cond_if_ne_null(Node *_Nullable head) {
    Node *p;
    if ((p = assign_get_next(head)) != nullptr) { // expected-warning{{passing nullable pointer to nonnull parameter}} expected-note{{add a null check before the call}}
        p->value = 1; // OK -- p narrowed
    }
}

void assign_cond_for_ne_null(Node *_Nullable head) {
    for (Node *p; (p = assign_get_next(head)) != nullptr;) { // expected-warning{{passing nullable pointer to nonnull parameter}} expected-note{{add a null check before the call}}
        p->value = 1; // OK -- p narrowed
    }
}

#pragma clang assume_nonnull end

// ===----------------------------------------------------------------------===//
// Boolean intermediary narrowing
// ===----------------------------------------------------------------------===//

#pragma clang assume_nonnull begin

void bool_test_ne_nullptr(Node * _Nullable p) {
    bool valid = (p != nullptr);
    if (valid) {
        (void)p->value; // OK
    }
    // Outside the if, p is still nullable
    (void)p->value; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

void bool_test_eq_nullptr_negated(Node * _Nullable p) {
    bool isNull = (p == nullptr);
    if (!isNull) {
        (void)p->value; // OK
    }
}

void bool_test_pointer_truthiness(Node * _Nullable p) {
    bool valid = p;
    if (valid) {
        (void)p->value; // OK
    }
}

void bool_test_negated_pointer(Node * _Nullable p) {
    bool isNull = !p;
    if (!isNull) {
        (void)p->value; // OK
    }
}

// Invalidation

void bool_test_pointer_reassigned(Node * _Nullable p, Node * _Nullable q) {
    bool valid = (p != nullptr);
    p = q; // reassign pointer -- bool guard is stale
    if (valid) {
        (void)p->value; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
    }
}

void bool_test_bool_reassigned(Node * _Nullable p) {
    bool valid = (p != nullptr);
    valid = false;
    if (valid) {
        (void)p->value; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
    }
}

void bool_test_pointer_incremented(int * _Nullable p) {
    bool valid = (p != nullptr);
    p++; // expected-warning{{pointer arithmetic on nullable pointer}} expected-note{{add a null check before performing arithmetic}}
    if (valid) {
        (void)*p; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
    }
}

// Negated conjunction

void bool_test_negated_and_return(Node * _Nullable p, Node * _Nullable q) {
    if (!(p && q)) return;
    (void)p->value; // OK
    (void)q->value; // OK
}

void bool_test_negated_and_else(Node * _Nullable p, Node * _Nullable q) {
    if (!(p && q)) {
        (void)p->value; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
    } else {
        (void)p->value; // OK
        (void)q->value; // OK
    }
}

void bool_test_negated_triple_and(Node * _Nullable a, Node * _Nullable b, Node * _Nullable c) {
    if (!(a && b && c)) return;
    (void)a->value; // OK
    (void)b->value; // OK
    (void)c->value; // OK
}

void bool_test_negated_and_ne_nullptr(Node * _Nullable p, Node * _Nullable q) {
    if (!(p != nullptr && q != nullptr)) return;
    (void)p->value; // OK
    (void)q->value; // OK
}

// Combined: bool guard + negated &&

void bool_test_guard_in_and(Node * _Nullable p, Node * _Nullable q) {
    bool pOk = (p != nullptr);
    if (pOk && q) {
        (void)p->value; // OK
        (void)q->value; // OK
    }
}

// Bool guard does not track compound conditions

void bool_test_compound_not_tracked(Node * _Nullable p, Node * _Nullable q) {
    bool both = (p && q);
    if (both) {
        // Compound conditions are not decomposed into per-variable guards
        (void)p->value; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
        (void)q->value; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
    }
}

#pragma clang assume_nonnull end

// ===----------------------------------------------------------------------===//
// Brace-wrapped assertion macros
// ===----------------------------------------------------------------------===//

#define INVARIANT(cond) \
  { \
    if (!(cond)) { \
      abort_handler("invariant failed"); \
    } \
  }

#define INVARIANT_MSG(cond, msg) \
  { \
    if (!(cond)) { \
      abort_handler(msg); \
    } \
  }

#pragma clang assume_nonnull begin

void brace_test_basic(Node* _Nullable p) {
    INVARIANT(p);
    p->value = 1; // OK - INVARIANT ensures p is non-null
}

void brace_test_with_message(Node* _Nullable p) {
    INVARIANT_MSG(p, "p must not be null");
    p->value = 1; // OK
}

void brace_test_ne_nullptr(Node* _Nullable p) {
    INVARIANT(p != nullptr);
    p->value = 1; // OK - p != nullptr checked
}

void brace_test_multiple_vars(Node* _Nullable p, Node* _Nullable q) {
    INVARIANT(p);
    INVARIANT(q);
    p->value = q->value; // OK - both narrowed
}

void brace_test_member(Node* _Nullable p) {
    INVARIANT(p);
    INVARIANT(p->next);
    p->next->value = 1; // OK - both p and p->next narrowed through macro
}

void brace_test_no_assert_still_warns(Node* _Nullable p) {
    p->value = 1; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

void brace_test_manual_bare_noreturn(Node* _Nullable p) {
    {
        if (!p) {
            abort_handler("null");
        }
    }
    p->value = 1; // OK - bare braces with noreturn narrow outward
}

void brace_test_nested(Node* _Nullable p, Node* _Nullable q) {
    {
        if (!p) { abort_handler("p"); }
        if (!q) { abort_handler("q"); }
    }
    p->value = q->value; // OK - both narrowed
}

void brace_test_does_not_affect_unrelated(Node* _Nullable p, Node* _Nullable q) {
    INVARIANT(p);
    q->value = 1; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

struct Widget {
    Node* _Nullable data;
    int x;

    void test_this_arrow() {
        this->x = 1; // OK - 'this' is never null
    }

    int test_this_deref() {
        return (*this).x; // OK - 'this' is never null
    }

    void test_this_member_narrowing() {
        INVARIANT(data);
        data->value = 1; // OK - data narrowed by INVARIANT
    }

    void test_this_member_if_narrowing() {
        if (data) {
            data->value = 1; // OK - data narrowed by if
        }
    }

    void test_this_member_no_narrowing() {
        data->value = 1; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
    }
};

// this->member narrowing must survive reassignment from nullable source.
// Regression: NullableThisMembers wasn't cleared when NarrowedThisMembers
// was set, so a post-reassignment null check was ignored.
struct TreeWalker {
    Node * _Nullable sel_ = nullptr;

    void reassign_then_standalone_if() {
        if (sel_ == nullptr) return;
        sel_ = sel_->parent; // sel_ becomes nullable (parent is _Nullable)
        if (sel_ != nullptr) {
            sel_->value = 1; // OK - re-narrowed by if
        }
    }

    void reassign_then_short_circuit_and() {
        if (sel_ == nullptr) return;
        sel_ = sel_->parent;
        if (sel_ != nullptr && sel_->value > 0) { // OK - && narrows sel_
            sel_->value = 1; // OK - still narrowed inside body
        }
    }

    void reassign_then_short_circuit_in_else_if(int cmd) {
        if (sel_ == nullptr) return;
        if (cmd == 1) {
            sel_ = sel_->parent;
        } else if (cmd == 2) {
            if (sel_ != nullptr && sel_->value > 0) { // OK
                sel_->value = 1; // OK
            }
        }
    }

    void reassign_no_recheck() {
        if (sel_ == nullptr) return;
        sel_ = sel_->parent;
        sel_->value = 1; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
    }
};

void brace_test_and_member_narrowing(Node* _Nullable p) {
    if (p && p->next) {
        p->next->value = 1; // OK - both p and p->next narrowed by && condition
    }
}

void brace_test_and_member_no_narrowing(Node* _Nullable p) {
    if (p) {
        p->next->value = 1; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
    }
}

void brace_test_or_member_early_return(Node* _Nullable p) {
    if (!p || !p->next) return;
    p->next->value = 1; // OK - both p and p->next narrowed by early return
}

void localvar_member_and_decomposition(Node* _Nullable p, int cmd) {
    if (cmd == 1) {
        if (p != nullptr && p->next != nullptr && p->next->value > 0) {
            p->next->value = 2; // OK - p->next narrowed by && in if-body
        }
    } else if (cmd == 2 && p != nullptr && p->next != nullptr) {
        p->next->value = 3; // OK - narrowed through else-if &&
    }
}

#pragma clang assume_nonnull end

// ===----------------------------------------------------------------------===//
// __builtin_expect and assertion macros
// ===----------------------------------------------------------------------===//

#pragma clang assume_nonnull begin

void builtin_test_expect_if(Entity* _Nullable p) {
    if (__builtin_expect(!!(p), 1)) {
        p->x = 1; // OK - narrowed through __builtin_expect
    }
}

void builtin_test_expect_negated(Entity* _Nullable p) {
    if (__builtin_expect(!!(p == nullptr), 0))
        return;
    p->x = 1; // OK - early return narrowing through __builtin_expect
}

void builtin_test_expect_early_return(Entity* _Nullable p) {
    if (__builtin_expect(!!(!p), 0))
        return;
    p->x = 1; // OK
}

#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)

void builtin_test_likely_macro(Entity* _Nullable p) {
    if (LIKELY(p)) {
        p->x = 1; // OK - narrowed
    }
}

void builtin_test_unlikely_null_check(Entity* _Nullable p) {
    if (UNLIKELY(!p))
        return;
    p->x = 1; // OK
}

#define CHECK(cond) do { if (__builtin_expect(!(cond), 0)) fatal("CHECK failed"); } while(0)

void builtin_test_check_macro(Entity* _Nullable p) {
    CHECK(p);
    p->x = 1; // OK - CHECK asserted non-null
}

void builtin_test_check_macro_two_vars(Entity* _Nullable p, Entity* _Nullable q) {
    CHECK(p);
    CHECK(q);
    p->x = q->x; // OK
}

void builtin_test_assume_simple(Entity* _Nullable p) {
    __builtin_assume(p != nullptr);
    p->x = 1; // OK - narrowed by __builtin_assume
}

void builtin_test_assume_truthiness(Entity* _Nullable p) {
    __builtin_assume(p);
    p->x = 1; // OK - narrowed by __builtin_assume(p)
}

void builtin_test_assume_two_vars(Entity* _Nullable p, Entity* _Nullable q) {
    __builtin_assume(p != nullptr);
    __builtin_assume(q != nullptr);
    p->x = q->x; // OK
}

void builtin_test_no_narrowing_without_check(Entity* _Nullable p) {
    p->x = 1; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

#pragma clang assume_nonnull end

// ===----------------------------------------------------------------------===//
// Cast propagation
// ===----------------------------------------------------------------------===//

struct Base {
    int x;
};

struct Derived : Base {
    int y;
};

Base * _Nullable getCastNullable();
Base * _Nonnull getCastNonnull();

#pragma clang assume_nonnull begin

void cast_test_c_style_nonnull() {
    Base *b = getCastNonnull();
    Derived *d = (Derived *)b;
    d->y = 1; // OK - nonnull propagated through C-style cast
}

void cast_test_static_cast_nonnull() {
    Base *b = getCastNonnull();
    Derived *d = static_cast<Derived *>(b);
    d->y = 1; // OK - nonnull propagated through static_cast
}

void cast_test_reinterpret_cast_nonnull() {
    Base *b = getCastNonnull();
    int *ip = reinterpret_cast<int *>(b);
    *ip = 1; // OK - nonnull propagated through reinterpret_cast
}

void cast_test_c_style_nullable_warns() {
    Base *b = getCastNullable();
    Derived *d = (Derived *)b;
    d->y = 1; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

void cast_test_explicit_nonnull_dest() {
    Base *b = getCastNullable();
    Derived * _Nonnull d = (Derived * _Nonnull)b;
    d->y = 1; // OK - explicit _Nonnull on dest type
}

// reinterpret_cast on this + pointer arithmetic

struct CastFoo {
    int x;
    void test_cast_this() {
        auto* p = reinterpret_cast<uint8_t*>(this) + 4;
        *p = 0; // OK -- this is always non-null, arithmetic preserves it
    }
};

void cast_test_ptr_arith_nonnull(int* p) {
    auto* q = p + 1;
    *q = 0; // OK -- p is nonnull (assume_nonnull), arithmetic preserves it
}

void cast_test_ptr_arith_nullable(int* _Nullable p) {
    auto* q = p + 1; // expected-warning{{pointer arithmetic on nullable pointer}} expected-note{{add a null check before performing arithmetic}}
    *q = 0; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

void ptr_diff_both_checked(int *_Nullable p, int *_Nonnull q) {
    auto d1 = p - q; // expected-warning{{pointer arithmetic on nullable pointer}} expected-note{{add a null check}}
    auto d2 = q - p; // expected-warning{{pointer arithmetic on nullable pointer}} expected-note{{add a null check}}
}

void ptr_diff_both_nonnull(int *_Nonnull p, int *_Nonnull q) {
    auto d = p - q; // no warning
}

void ptr_diff_after_check(int *_Nullable p, int *_Nonnull q) {
    if (p) {
        auto d = p - q; // no warning — checked
    }
}

void ptr_arith_all_forms(int *_Nullable p) {
    p++;  // expected-warning{{pointer arithmetic on nullable pointer}} expected-note{{add a null check}}
    p--;  // expected-warning{{pointer arithmetic on nullable pointer}} expected-note{{add a null check}}
    p += 1; // expected-warning{{pointer arithmetic on nullable pointer}} expected-note{{add a null check}}
    p -= 1; // expected-warning{{pointer arithmetic on nullable pointer}} expected-note{{add a null check}}
}

void ptr_arith_nonnull_no_warn(int *_Nonnull p) {
    p++;    // no warning
    p--;    // no warning
    p += 1; // no warning
    auto *q = p + 1; // no warning
}

struct DerivedReinterpret : Base {
    void test_reinterpret_cast_this_to_base() {
        Base *b = reinterpret_cast<Base*>(this);
        b->x = 1; // OK -- this is non-null

        uint8_t *raw = reinterpret_cast<uint8_t*>(this) + 4;
        *raw = 0; // OK -- this is always non-null
    }
};

struct Bar {
    int val;
    void test_deref_static_cast_this() {
        (*static_cast<Bar*>(this)).val = 42; // OK -- this is non-null
    }
};

struct DerivedBar : Base {
    void test_deref_cast_this_to_base() {
        (*static_cast<Base*>(this)).x = 1; // OK -- this is non-null
    }
};

struct Baz {
    int z;
};

void cast_test_deref_cast_addr_of(Baz& other) {
    (*static_cast<Baz*>(&other)).z = 1; // OK -- address-of is non-null
}

void cast_test_deref_cast_addr_of_different_type(Derived& d) {
    (*static_cast<Base*>(&d)).x = 1; // OK -- address-of is non-null
}

#pragma clang assume_nonnull end

// ===----------------------------------------------------------------------===//
// Chained and nested dereferences
// ===----------------------------------------------------------------------===//

#pragma clang assume_nonnull begin

void chain_test_direct_warns() {
    int v = getNode()->value; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
    (void)v;
}

void chain_test_direct_guarded() {
    Node * _Nullable n = getNode();
    if (n) {
        int v = n->value; // OK
        (void)v;
    }
}

void chain_test_double_warns() {
    (void)getContainer()->root; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

void chain_test_double_partial_guard() {
    Container * _Nullable c = getContainer();
    if (c) {
        c->root->value = 1; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
    }
}

void chain_test_double_full_guard() {
    Container * _Nullable c = getContainer();
    if (c && c->root) {
        c->root->value = 1; // OK -- both narrowed
    }
}

void chain_test_triple(Node * _Nullable head) {
    if (head && head->next && head->next->next) {
        head->next->next->value = 42; // OK -- all three narrowed
    }
}

// Known limitation: multi-level member narrowing
void chain_test_triple_partial(Node * _Nullable head) {
    if (head && head->next) {
        // head->next is narrowed, but head->next->next is still nullable.
        // The analysis currently does not warn here (accepted false negative).
        head->next->next->value = 42; // no warning (known limitation)
    }
}

// Method return chaining

struct Builder {
    Node * _Nullable node;

    Builder * _Nullable setNode(Node * _Nonnull n) {
        node = n;
        return this;
    }

    Node * _Nullable getResult() {
        return node;
    }
};

Builder * _Nullable getBuilder();

void chain_test_builder_warns() {
    getBuilder()->getResult(); // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

void chain_test_builder_guarded() {
    Builder * _Nullable b = getBuilder();
    if (b) {
        Node * _Nullable result = b->getResult();
        if (result) {
            (void)result->value; // OK -- both guarded
        }
    }
}

// Pointer-to-pointer (T**) -- known limitation
void chain_test_ptr_to_ptr(Node * _Nullable * _Nullable pp) {
    if (pp && *pp) {
        (*pp)->value = 1; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
    }
}

void chain_test_ptr_to_ptr_via_local(Node * _Nullable * _Nullable pp) {
    if (!pp) return;
    Node * _Nullable p = *pp;
    if (p) {
        p->value = 1; // OK -- local variable is tracked
    }
}

void chain_test_array_of_nullable(Node * _Nullable nodes[], int n) {
    for (int i = 0; i < n; i++) {
        Node * _Nullable cur = nodes[i];
        if (cur) {
            cur->value = i; // OK -- narrowed via local
        }
    }
}

void chain_test_conditional(Node * _Nullable p) {
    Node * _Nullable next = p ? p->next : nullptr;
    if (next) {
        next->value = 1; // OK -- narrowed
    }
}

void chain_test_assign_from() {
    Node * _Nullable n = getNode();
    if (!n) return;
    Node * _Nullable child = n->next;
    if (child) {
        child->value = 1; // OK
    }
}

void chain_test_in_loop() {
    Node * _Nullable head = getNode();
    for (Node * _Nullable p = head; p; p = p->next) {
        if (p->left && p->left->right) {
            p->left->right->value = 0; // OK -- all narrowed
        }
    }
}

Node * _Nullable chain_get_grandchild(Node * _Nullable n) {
    if (n && n->next) {
        return n->next->next; // OK -- n->next narrowed; returns nullable
    }
    return nullptr;
}

void chain_test_container_accessor() {
    Container * _Nullable c = getContainer();
    if (!c) return;
    if (!c->root) return;
    c->root->value = 1; // OK -- both narrowed

    if (c->root->next) {
        c->root->next->value = 2; // OK
    }
}

Node * _Nullable chain_safe_next(Node * _Nullable n) {
    if (!n) return nullptr;
    return n->next; // OK -- n narrowed
}

void chain_test_cascade() {
    Node * _Nullable n = getNode();
    Node * _Nullable child = chain_safe_next(n);
    if (child) {
        child->value = 1; // OK
    }
}

struct Tree {
    int data;
    Tree * _Nullable left;
    Tree * _Nullable right;
    Tree * _Nullable parent;
};

void chain_test_tree_traversal(Tree * _Nullable root) {
    if (!root) return;
    if (root->left) {
        root->left->data = 1; // OK
        if (root->left->left) {
            root->left->left->data = 2; // OK -- deeply narrowed
        }
    }
    if (root->right && root->right->parent) {
        root->right->parent->data = 3; // OK
    }
}

void chain_test_invalidation(Node * _Nullable p) {
    if (p && p->next) {
        p->next->value = 1; // OK -- both narrowed
        p = getNode();       // reassign p -- narrowing gone
        if (p) {
            p->value = 2; // OK -- re-narrowed
        }
    }
}

#pragma clang assume_nonnull end

// ===----------------------------------------------------------------------===//
// Complex CFG patterns (diamonds, loops, merges)
// ===----------------------------------------------------------------------===//

Node * _Nonnull getSafeNode();

#pragma clang assume_nonnull begin

void cfg_test_diamond_both_narrow(Node * _Nullable p) {
    if (getInt()) {
        if (!p) return;
    } else {
        if (!p) return;
    }
    (void)p->value; // OK -- narrowed on both paths
}

void cfg_test_diamond_one_narrows(Node * _Nullable p) {
    if (getInt()) {
        if (!p) return;
    } else {
        // NOT narrowed here
    }
    (void)p->value; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

void cfg_test_diamond_assign_both(int * _Nullable p) {
    int x = 0, y = 0;
    if (getInt()) {
        p = &x;
    } else {
        p = &y;
    }
    (void)*p; // OK -- both branches assign nonnull (address-of)
}

void cfg_test_diamond_assign_one(Node * _Nullable p) {
    int x;
    if (getInt()) {
        // p unchanged -- still nullable
    } else {
        if (!p) return;
    }
    (void)p->value; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

void cfg_test_nested_three_levels(Node * _Nullable p, Node * _Nullable q) {
    if (!p) return;
    if (getInt()) {
        if (!q) return;
        (void)p->value; // OK
        (void)q->value; // OK
    } else {
        (void)p->value; // OK -- outer guard still holds
        (void)q->value; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
    }
}

void cfg_test_loop_multi_exit(Node * _Nullable p) {
    for (int i = 0; i < 10; i++) {
        if (!p) break;
        (void)p->value; // OK -- narrowed by break guard
    }
    (void)p->value; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

void cfg_test_sequential_narrow(Node * _Nullable p, Node * _Nullable q, Node * _Nullable r) {
    if (!p) return;
    if (!q) return;
    if (!r) return;
    (void)p->value; // OK
    (void)q->value; // OK
    (void)r->value; // OK
}

void cfg_test_reassign_in_branch(Node * _Nullable p) {
    if (!p) return;
    if (getInt()) {
        p = getNode(); // reassigned to nullable
    }
    (void)p->value; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

void cfg_test_do_while(Node * _Nullable p) {
    if (!p) return;
    do {
        (void)p->value; // OK -- narrowed on entry
    } while (getInt() && p);
}

void cfg_test_nested_loops(Node * _Nullable p) {
    if (!p) return;
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            (void)p->value; // OK -- narrowed, loops don't invalidate
        }
    }
}

void cfg_test_switch_fallthrough(Node * _Nullable p) {
    switch (getInt()) {
    case 0:
        if (!p) return;
        [[fallthrough]];
    case 1:
        // Reached from case 0 (narrowed) OR case 1 (not narrowed)
        (void)p->value; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
        break;
    default:
        break;
    }
}

void cfg_test_post_loop_narrowing(Node * _Nullable p) {
    while (true) {
        if (p) break;
        p = getNode();
    }
    (void)p->value; // OK -- only exit is via break where p is narrowed
}

Node * _Nonnull cfg_test_both_return(Node * _Nullable p) {
    if (p) {
        return p; // OK
    } else {
        return getSafeNode();
    }
}

void cfg_test_ternary_chain(Node * _Nullable a, Node * _Nullable b, Node * _Nullable c) {
    Node *picked = a ? a : (b ? b : c);
    if (picked)
        (void)picked->value; // OK -- narrowed
}

void cfg_test_while_reassign(Node * _Nullable p) {
    while (p) {
        (void)p->value; // OK -- narrowed by while condition
        p = p->left;
    }
}

#pragma clang assume_nonnull end

// ===----------------------------------------------------------------------===//
// Compound conditions (&&, ||, De Morgan)
// ===----------------------------------------------------------------------===//

bool isValid(Node * _Nonnull p);

#pragma clang assume_nonnull begin

void compound_test_and_both(Node * _Nullable p, Node * _Nullable q) {
    if (p && q) {
        (void)p->value; // OK
        (void)q->value; // OK
    }
}

void compound_test_or_neither(Node * _Nullable p, Node * _Nullable q) {
    if (p || q) {
        (void)p->value; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
        (void)q->value; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
    }
}

void compound_test_demorgan_return(Node * _Nullable p, Node * _Nullable q) {
    if (!p || !q) return;
    (void)p->value; // OK
    (void)q->value; // OK
}

void compound_test_chain(Node * _Nullable p) {
    if (p && p->next) {
        (void)p->value; // OK
        (void)p->next->value; // OK
    }
}

void compound_test_triple_and(Node * _Nullable a, Node * _Nullable b, Node * _Nullable c) {
    if (a && b && c) {
        (void)a->value; // OK
        (void)b->value; // OK
        (void)c->value; // OK
    }
}

void compound_test_negated_and(Node * _Nullable p, Node * _Nullable q) {
    if (!(p && q)) return;
    (void)p->value; // OK
    (void)q->value; // OK
}

void compound_test_ne_null_and(Node * _Nullable p, Node * _Nullable q) {
    if (p != nullptr && q != nullptr) {
        (void)p->value; // OK
        (void)q->value; // OK
    }
}

void compound_test_eq_null_or_return(Node * _Nullable p, Node * _Nullable q) {
    if (p == nullptr || q == nullptr) return;
    (void)p->value; // OK
    (void)q->value; // OK
}

void compound_test_mixed_condition(Node * _Nullable p) {
    if (p && p->value > 0) {
        (void)p->value; // OK
    }
}

void compound_test_condition_with_call(Node * _Nullable p) {
    if (p && isValid(p)) {
        (void)p->value; // OK
    }
}

void compound_test_while_and(Node * _Nullable p) {
    while (p && p->next) {
        (void)p->value; // OK
        p = p->next;
    }
}

void compound_test_for_and_condition(Node * _Nullable p) {
    for (int i = 0; p && i < 10; i++) {
        (void)p->value; // OK -- narrowed by for condition
    }
}

void compound_test_ternary_null_check(Node * _Nullable p) {
    int v = p ? p->value : -1; // OK -- p narrowed in true branch
}

void compound_test_multi_ternary(Node * _Nullable a, Node * _Nullable b) {
    int v = a ? a->value : (b ? b->value : 0); // OK
}

void compound_test_bool_intermediary(Node * _Nullable p) {
    bool valid = (p != nullptr);
    if (valid) {
        (void)p->value; // OK
    }
}

void compound_test_bool_eq_null(Node * _Nullable p) {
    bool isNull = (p == nullptr);
    if (!isNull) {
        (void)p->value; // OK
    }
}

void compound_test_bool_truthiness(Node * _Nullable p) {
    bool valid = p;
    if (valid) {
        (void)p->value; // OK
    }
}

void compound_test_bool_negated_ptr(Node * _Nullable p) {
    bool isNull = !p;
    if (!isNull) {
        (void)p->value; // OK
    }
    if (isNull) {
        (void)p->value; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
    }
}

void compound_test_bool_ptr_reassigned(Node * _Nullable p, Node * _Nullable q) {
    bool valid = (p != nullptr);
    p = q;
    if (valid) {
        (void)p->value; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
    }
}

void compound_test_bool_reassigned(Node * _Nullable p) {
    bool valid = (p != nullptr);
    valid = false;
    if (valid) {
        (void)p->value; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
    }
}

void compound_test_negated_triple_and(Node * _Nullable a, Node * _Nullable b, Node * _Nullable c) {
    if (!(a && b && c)) return;
    (void)a->value; // OK
    (void)b->value; // OK
    (void)c->value; // OK
}

void compound_test_negated_and_body(Node * _Nullable p, Node * _Nullable q) {
    if (!(p && q)) {
        (void)p->value; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
    } else {
        (void)p->value; // OK
        (void)q->value; // OK
    }
}

#pragma clang assume_nonnull end

// ===----------------------------------------------------------------------===//
// Duplicate diagnostic suppression
// ===----------------------------------------------------------------------===//

#pragma clang assume_nonnull begin

void dupdiag_take_nonnull(int * _Nonnull p);

void dupdiag_test_pass_to_nonnull(int * _Nullable p) {
    dupdiag_take_nonnull(p); // expected-warning{{passing nullable pointer to nonnull parameter}} expected-note{{add a null check before the call}}
    *p = 42; // OK -- narrowed by nonnull call, no second warning
}

void dupdiag_test_deref_only(int * _Nullable p) {
    *p = 42; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

void dupdiag_test_assign_to_nonnull(int * _Nullable p) {
    int * _Nonnull q = p; // expected-warning{{assigning nullable pointer to nonnull variable}} expected-note{{add a null check before assigning}}
}

void dupdiag_test_checked(int * _Nullable p) {
    if (!p) return;
    dupdiag_take_nonnull(p); // OK -- narrowed, no warning
    *p = 42;                 // OK -- narrowed, no deref warning
}

#pragma clang assume_nonnull end

// ===----------------------------------------------------------------------===//
// Else-branch narrowing
// ===----------------------------------------------------------------------===//

#pragma clang assume_nonnull begin

void else_test_simple(Entity* _Nullable p) {
    if (!p) {
        return;
    } else {
        p->x = 1; // OK - narrowed in else branch
    }
}

void else_test_or_two_vars(Entity* _Nullable p, Entity* _Nullable q) {
    if (!p || !q) {
        return;
    } else {
        p->x = q->x; // OK - both narrowed in else branch
    }
}

void else_test_or_three_vars(Entity* _Nullable p, Entity* _Nullable q, Entity* _Nullable r) {
    if (!p || !q || !r) {
        return;
    } else {
        p->x = q->x + r->x; // OK - all narrowed in else branch
    }
}

void else_test_early_return_or(Entity* _Nullable p, Entity* _Nullable q) {
    if (!p || !q)
        return;
    p->x = q->x; // OK - both narrowed after early return
}

void else_test_positive_no_narrow(Entity* _Nullable p) {
    if (p) {
        p->x = 1; // OK - narrowed in then branch
    } else {
        p->x = 2; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
    }
}

void else_test_member_narrowing(Entity* _Nullable p) {
    if (!p) {
        // p is null here
    } else {
        p->x = 1; // OK - narrowed in else
    }
}

#pragma clang assume_nonnull end

// ===----------------------------------------------------------------------===//
// For-loop narrowing
// ===----------------------------------------------------------------------===//

#pragma clang assume_nonnull begin

void for_test_linked_list(Node* _Nullable head) {
    for (Node* _Nullable p = head; p; p = p->next) {
        p->value = 0; // OK - p narrowed from condition
    }
}

void for_test_simple_increment(Node* _Nullable p) {
    for (; p; p = p->next) {
        p->value = 0; // OK
    }
}

#pragma clang assume_nonnull end

// ===----------------------------------------------------------------------===//
// Nested if-init (regression test for && narrowing at IfStmt merge)
// ===----------------------------------------------------------------------===//

template<typename K, typename V> struct DenseMap {
    struct Iter { V second; bool operator!=(Iter o) const; };
    Iter find(K key) const;
    Iter end() const;
};

struct VarDecl {
    struct QT { bool isPointerType() const; bool isBooleanType() const; };
    QT getType() const;
};

using BoolGuardMap = DenseMap<const VarDecl *, int>;

struct Expr {};
template<typename T, typename U> T *dyn_cast(U *);

void nested_if_init_and_narrowing(const Expr * _Nullable E, const BoolGuardMap * _Nullable BoolGuards) {
    if (auto *VD = dyn_cast<VarDecl, const Expr>(E)) {
        if (VD->getType().isPointerType())
            return;
        if (BoolGuards && VD->getType().isBooleanType()) {
            BoolGuards->find(VD); // OK -- BoolGuards narrowed by &&
            (void)BoolGuards->end(); // OK
        }
    }
}

// ===----------------------------------------------------------------------===//
// __attribute__((nonnull)) interactions
// ===----------------------------------------------------------------------===//

#pragma clang assume_nonnull begin

__attribute__((nonnull))
void nonnull_attr_consume_all(Node *a, Node *b) {}

void nonnull_attr_test_fn_level(Node * _Nullable p, Node * _Nullable q) {
    if (!p || !q) return;
    nonnull_attr_consume_all(p, q); // OK -- both narrowed
    (void)p->value; // OK
    (void)q->value; // OK
}

__attribute__((nonnull(1, 3)))
void nonnull_attr_consume_specific(Node *a, Node * _Nullable b, Node *c) {}

void nonnull_attr_test_param_level(Node * _Nullable p, Node * _Nullable q, Node * _Nullable r) {
    nonnull_attr_consume_specific(p, q, r); // expected-warning 2{{passing nullable pointer to nonnull parameter}} expected-note 2{{add a null check before the call}}
    (void)p->value; // OK -- narrowed by passing to nonnull param 1
    (void)r->value; // OK -- narrowed by passing to nonnull param 3
}

__attribute__((returns_nonnull))
Node *nonnull_attr_createSafe();

void nonnull_attr_test_returns_nonnull() {
    Node *p = nonnull_attr_createSafe();
    (void)p->value; // OK -- _Nonnull return type
}

void nonnull_attr_take_nonnull(Node * _Nonnull p) {}

void nonnull_attr_test_type_qualifier(Node * _Nullable p) {
    nonnull_attr_take_nonnull(p); // expected-warning{{passing nullable pointer to nonnull parameter}} expected-note{{add a null check before the call}}
    (void)p->value; // OK -- narrowed by passing to _Nonnull param
}

void nonnull_attr_test_multi_call(Node * _Nullable a, Node * _Nullable b) {
    nonnull_attr_take_nonnull(a); // expected-warning{{passing nullable pointer to nonnull parameter}} expected-note{{add a null check before the call}}
    nonnull_attr_take_nonnull(b); // expected-warning{{passing nullable pointer to nonnull parameter}} expected-note{{add a null check before the call}}
    (void)a->value; // OK
    (void)b->value; // OK
}

void nonnull_attr_unrelated_fn();

void nonnull_attr_test_survives_calls(Node * _Nonnull p) {
    nonnull_attr_unrelated_fn();
    (void)p->value; // OK -- _Nonnull parameter, calls don't invalidate
}

extern "C" {
    __attribute__((nonnull(1)))
    void nonnull_attr_c_consumer(Node *p, int x);
}

void nonnull_attr_test_c_fn(Node * _Nullable p) {
    nonnull_attr_c_consumer(p, 42); // expected-warning{{passing nullable pointer to nonnull parameter}} expected-note{{add a null check before the call}}
    (void)p->value; // OK -- narrowed by passing to nonnull param
}

#pragma clang assume_nonnull end

// ===----------------------------------------------------------------------===//
// _Nonnull parameter tests (no diagnostics expected)
// ===----------------------------------------------------------------------===//

#pragma clang assume_nonnull begin

void nonnull_param_test_star(Entity* _Nonnull p) {
    (*p).x = 1; // OK - _Nonnull never warns
}

void nonnull_param_test_arrow(Entity* _Nonnull p) {
    p->x = 1; // OK - _Nonnull never warns
}

void nonnull_param_test_method(Entity* _Nonnull p) {
    int v = p->value(); // OK
}

void nonnull_param_test_local() {
    Entity e;
    Entity* _Nonnull p = &e;
    p->x = 1; // OK - _Nonnull local
}

void nonnull_param_test_mixed(Entity* _Nonnull safe, Entity* _Nullable risky) {
    safe->x = 1; // OK - _Nonnull
    if (risky) {
        risky->x = safe->x; // OK - risky narrowed, safe is _Nonnull
    }
}

void nonnull_param_test_after_null_check(Entity* _Nonnull p) {
    if (p) {
        p->x = 1; // OK - redundant check, but still fine
    }
    p->x = 2; // OK - _Nonnull
}

#pragma clang assume_nonnull end

// ===----------------------------------------------------------------------===//
// _Nonnull parameter narrowing (passing nullable to nonnull)
// ===----------------------------------------------------------------------===//

// Simulate system header declarations with _Nonnull params
size_t my_strlen(const char * _Nonnull s);
void my_use(const char * _Nonnull s);
void unannotated_use(const char *s);
void two_params(const char * _Nonnull a, const char *b);

// GCC-style nonnull attribute
size_t gcc_strlen(const char *s) __attribute__((nonnull(1)));
void gcc_all_nonnull(const char *a, const char *b) __attribute__((nonnull));
void gcc_partial_nonnull(const char *a, const char *b) __attribute__((nonnull(1)));

#pragma clang assume_nonnull begin

void narrow_param_test_nonnull(const char * _Nullable filePath) {
    my_strlen(filePath); // expected-warning{{passing nullable pointer to nonnull parameter}} expected-note{{add a null check before the call}}
    const char c = *filePath; // OK -- narrowed by call above
}

void narrow_param_test_unannotated_no_narrow(const char * _Nullable filePath) {
    unannotated_use(filePath);
    const char c = *filePath; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

void narrow_param_test_mixed(const char * _Nullable a, const char * _Nullable b) {
    two_params(a, b); // expected-warning{{passing nullable pointer to nonnull parameter}} expected-note{{add a null check before the call}}
    const char c1 = *a; // OK -- narrowed
    const char c2 = *b; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

void narrow_param_test_multiple_calls(const char * _Nullable p, const char * _Nullable q) {
    my_use(p); // expected-warning{{passing nullable pointer to nonnull parameter}} expected-note{{add a null check before the call}}
    my_strlen(q); // expected-warning{{passing nullable pointer to nonnull parameter}} expected-note{{add a null check before the call}}
    const char c1 = *p; // OK
    const char c2 = *q; // OK
}

void narrow_param_test_gcc_nonnull(const char * _Nullable filePath) {
    gcc_strlen(filePath); // expected-warning{{passing nullable pointer to nonnull parameter}} expected-note{{add a null check before the call}}
    const char c = *filePath; // OK -- narrowed by gcc nonnull attr
}

void narrow_param_test_gcc_all(const char * _Nullable a, const char * _Nullable b) {
    gcc_all_nonnull(a, b); // expected-warning 2{{passing nullable pointer to nonnull parameter}} expected-note 2{{add a null check before the call}}
    const char c1 = *a; // OK -- narrowed
    const char c2 = *b; // OK -- narrowed
}

void narrow_param_test_gcc_partial(const char * _Nullable a, const char * _Nullable b) {
    gcc_partial_nonnull(a, b); // expected-warning{{passing nullable pointer to nonnull parameter}} expected-note{{add a null check before the call}}
    const char c1 = *a; // OK -- narrowed (param 1 is nonnull)
    const char c2 = *b; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

#pragma clang assume_nonnull end

// ===----------------------------------------------------------------------===//
// Noreturn, if-else termination, do-while assertions
// ===----------------------------------------------------------------------===//

#pragma clang assume_nonnull begin

void noreturn_test_if_else_both_return(Entity* _Nullable p) {
    if (!p) {
        if (true) { return; }
        else { return; }
    }
    p->x = 1; // OK - if always terminates
}

void noreturn_test_if_else_return_and_noreturn(Entity* _Nullable p) {
    if (!p) {
        if (true) { return; }
        else { fatal("unreachable"); }
    }
    p->x = 1; // OK
}

void noreturn_test_nested_if_else(Entity* _Nullable p) {
    if (!p) {
        if (true) {
            if (true) { return; }
            else { return; }
        } else {
            fatal("unreachable");
        }
    }
    p->x = 1; // OK - deeply nested, both paths terminate
}

void noreturn_test_if_without_else(Entity* _Nullable p, bool flag) {
    if (!p) {
        if (flag) { return; }
    }
    p->x = 1; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

void noreturn_test_function(Entity* _Nullable p) {
    if (!p) {
        fatal("p is null");
    }
    p->x = 1; // OK - noreturn guarantees we don't reach here if p was null
}

void noreturn_test_in_compound(Entity* _Nullable p) {
    if (!p) {
        log_msg("about to die");
        fatal("p is null");
    }
    p->x = 1; // OK
}

#define MY_ASSERT(cond) do { if (!(cond)) fatal("assertion failed: " #cond); } while(0)

void noreturn_test_do_while_assert(Entity* _Nullable p) {
    MY_ASSERT(p);
    p->x = 1; // OK - asserted non-null
}

void noreturn_test_do_while_assert_two_vars(Entity* _Nullable p, Entity* _Nullable q) {
    MY_ASSERT(p);
    MY_ASSERT(q);
    p->x = q->x; // OK - both asserted non-null
}

#pragma clang assume_nonnull end

// ===----------------------------------------------------------------------===//
// Reassignment invalidation
// ===----------------------------------------------------------------------===//

Entity* _Nullable getEntityForReassign();

#pragma clang assume_nonnull begin

void reassign_test_invalidates(Entity* _Nullable p, Entity* _Nullable other) {
    if (p) {
        p = other;
        (void)*p; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
    }
}

void reassign_test_deref_before(Entity* _Nullable p, Entity* _Nullable other) {
    if (p) {
        (*p).x = 1; // OK - narrowed
        p = other;
        (*p).x = 2; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
    }
}

void reassign_test_then_recheck(Entity* _Nullable p) {
    p = getEntityForReassign();
    if (p) {
        (*p).x = 1; // OK - re-narrowed after reassignment
    }
}

void reassign_test_increment_preserves(Entity* _Nullable p) {
    if (p) {
        p++;
        (void)*p; // OK -- p++ on non-null is still non-null
    }
}

void reassign_test_decrement_preserves(Entity* _Nullable p) {
    if (p) {
        --p;
        (void)*p; // OK -- --p on non-null is still non-null
    }
}

// Member narrowing IS invalidated by pointer arithmetic
struct Chain {
    int value;
    Chain * _Nullable next;
};

void reassign_test_increment_invalidates_member(Chain * _Nullable p) {
    if (p && p->next) {
        p++;
        p->next->value = 1; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
    }
}

#pragma clang assume_nonnull end

// ===----------------------------------------------------------------------===//
// Switch statement narrowing
// ===----------------------------------------------------------------------===//

#pragma clang assume_nonnull begin

void switch_test_before(Entity* _Nullable p, int kind) {
    if (!p) return;
    switch (kind) {
    case 0:
        p->x = 0; // OK - narrowed before switch
        break;
    case 1:
        p->x = 1; // OK - narrowing carries into cases
        break;
    default:
        p->x = -1; // OK
        break;
    }
}

void switch_test_null_check_then(Entity* _Nullable p, int kind) {
    if (p) {
        switch (kind) {
        case 0:
            p->x = 0; // OK
            break;
        case 1:
            p->x = 1; // OK
            break;
        }
    }
}

#pragma clang assume_nonnull end

// ===----------------------------------------------------------------------===//
// Terminators: throw, goto, break, continue
// ===----------------------------------------------------------------------===//

#pragma clang assume_nonnull begin

void term_test_throw_narrows(Entity* _Nullable p) {
    if (!p) throw "null pointer";
    p->x = 1; // OK - throw terminates
}

void term_test_throw_in_compound(Entity* _Nullable p) {
    if (!p) {
        throw "null";
    }
    p->x = 1; // OK
}

void term_test_goto_narrows(Entity* _Nullable p) {
    if (!p) goto cleanup;
    p->x = 1; // OK - goto terminates
cleanup:
    return;
}

void term_test_break_narrows(Entity* _Nullable p) {
    for (int i = 0; i < 10; i++) {
        if (!p) break;
        p->x = i; // OK - break terminates
    }
}

void term_test_break_while(Entity* _Nullable p) {
    while (true) {
        if (!p) break;
        p->x = 1; // OK
    }
}

void term_test_continue_narrows(Entity* _Nullable p) {
    for (int i = 0; i < 10; i++) {
        if (!p) continue;
        p->x = i; // OK - continue terminates
    }
}

void term_test_positive_check_else_return(Entity* _Nullable p) {
    if (p) {
        // use p
    } else {
        return;
    }
    p->x = 1; // OK - only reachable when p is non-null
}

void term_test_noreturn_then_deref(Entity* _Nullable p) {
    if (!p) fatal("null");
    p->x = 1; // OK
}

void term_test_two_checks_return(Entity* _Nullable p, Entity* _Nullable q) {
    if (!p) return;
    if (!q) return;
    p->x = q->x; // OK - both narrowed
}

#pragma clang assume_nonnull end

// ===----------------------------------------------------------------------===//
// Ternary operator narrowing
// ===----------------------------------------------------------------------===//

#pragma clang assume_nonnull begin

void ternary_test_true_branch(Node* _Nullable p) {
    int x = p ? p->value : 0; // OK - p narrowed to nonnull in true branch
}

void ternary_test_false_branch_negated(Node* _Nullable p) {
    int x = !p ? 0 : p->value; // OK - p narrowed to nonnull in false branch
}

void ternary_test_no_narrowing_false(Node* _Nullable p) {
    int x = p ? 0 : p->value; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

void ternary_test_deref_star(Node* _Nullable p) {
    Node n = p ? *p : (Node){0, nullptr, nullptr, nullptr, nullptr, nullptr}; // OK - p narrowed to nonnull
    (void)n;
}

void ternary_test_ne_null(Node* _Nullable p) {
    int x = (p != nullptr) ? p->value : -1; // OK
}

void ternary_test_eq_null(Node* _Nullable p) {
    int x = (p == nullptr) ? -1 : p->value; // OK - narrowed in false branch
}

void ternary_test_and_both(Node* _Nullable p, Node* _Nullable q) {
    int x = (p && q) ? p->value + q->value : 0; // OK - both narrowed
}

void ternary_test_nested(Node* _Nullable p, Node* _Nullable q) {
    int x = p ? (q ? p->value + q->value : p->value) : 0; // OK
}

void ternary_test_unrelated_cond(int flag, Node* _Nullable p) {
    int x = flag ? p->value : 0; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

#pragma clang assume_nonnull end

// ===----------------------------------------------------------------------===//
// Unannotated false positives (expected under nullable-default)
// ===----------------------------------------------------------------------===//

// Under -fnullability-default=nullable, unannotated pointers are nullable.
// These patterns correctly warn.

inline bool unannotated_getData(const uint8_t** buffers, int readIndex) {
    auto buffer = buffers[readIndex]; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
    return buffer != nullptr;
}

struct UnannotatedWidget {
    int x;
    ~UnannotatedWidget() {}
};

void unannotated_test_deleter() {
    auto* ptr = new UnannotatedWidget;
    auto deleter = [](UnannotatedWidget* w) {
        w->~UnannotatedWidget(); // OK — lambda params are auto-narrowed (callee trusts caller)
    };
    deleter(ptr);
}

// Lambda pointer params default to nonnull (auto-narrowed in body, verified
// at call sites). Explicit _Nullable overrides this default.
struct LambdaNode { int size; };
void lambda_param_auto_narrow() {
    // Comparator pattern: unannotated params are nonnull, no body warnings.
    auto cmp = [](const LambdaNode* a, const LambdaNode* b) -> bool {
        return a->size > b->size; // OK — lambda params default to nonnull
    };
    (void)cmp;

    // Captured pointers are NOT auto-narrowed — still nullable.
    LambdaNode* _Nullable np = nullptr;
    auto bad = [&]() {
        np->size; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
    };
    (void)bad;
}

// Call-site verification: passing nullable to a lambda warns.
void lambda_callsite_check(LambdaNode* _Nullable maybe) {
    auto work = [](LambdaNode* n) {
        n->size = 42; // OK — auto-narrowed
    };
    work(maybe); // expected-warning{{passing nullable pointer to nonnull parameter}} expected-note{{add a null check before the call}}

    LambdaNode* ok = new LambdaNode;
    work(ok); // OK — nonnull arg
}

// Explicit _Nullable lambda param: body warns on unchecked deref, call site
// does NOT warn when passing nullable (the param accepts null).
void lambda_explicit_nullable(LambdaNode* _Nullable maybe) {
    auto filter = [](LambdaNode* _Nullable n) {
        n->size; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
    };
    filter(maybe); // OK — param is _Nullable, accepts null
}

struct UnannotatedBuffer {
    int offset;
    uint8_t* getBuffer() {
        return reinterpret_cast<uint8_t*>(this) + offset;
    }
    void use() {
        uint8_t val = getBuffer()[0]; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
    }
};

// ===----------------------------------------------------------------------===//
// void* cast patterns
// ===----------------------------------------------------------------------===//

struct Data {
    int value;
};

void voidstar_test_cast_deref(void* obj) {
    Data* p = static_cast<Data*>(obj);
    *p = Data{42}; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
    p->value = 1;  // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

void voidstar_test_reinterpret_cast(void* obj) {
    *reinterpret_cast<void**>(obj) = nullptr; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

void voidstar_test_nullable(void* _Nullable obj) {
    Data* p = static_cast<Data*>(obj);
    *p = Data{42}; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

void voidstar_test_double_ptr(void* obj) {
    *reinterpret_cast<void**>(obj) = nullptr; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
    *reinterpret_cast<int**>(obj) = nullptr;  // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

void voidstar_test_double_ptr_local(void* obj) {
    void** pp = reinterpret_cast<void**>(obj);
    *pp = nullptr; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

void voidstar_test_checked(void* obj) {
    if (obj) {
        Data* p = static_cast<Data*>(obj);
        *p = Data{42}; // OK -- obj was checked
    }
}

// ===----------------------------------------------------------------------===//
// While-loop narrowing
// ===----------------------------------------------------------------------===//

#pragma clang assume_nonnull begin

void while_test_basic(Node* _Nullable p) {
    while (p) {
        p->value = 1; // OK - p narrowed by while condition
    }
}

void while_test_linked_list(Node* _Nullable head) {
    Node* _Nullable p = head;
    while (p) {
        p->value = 0;
        p = p->next; // OK - p narrowed, so p->next is safe
    }
}

void while_test_nested(Node* _Nullable p) {
    while (p) {
        Node* _Nullable q = p->next;
        while (q) {
            q->value = p->value; // OK - both narrowed
            q = q->next;
        }
        p = p->next;
    }
}

#pragma clang assume_nonnull end

// ===----------------------------------------------------------------------===//
// Range-for loop (works under both nullable and nonnull defaults)
// ===----------------------------------------------------------------------===//

struct Item { int value; };

template <typename T, int N>
struct Array {
    T data_[N];
    T* begin() { return data_; }
    T* end() { return data_ + N; }
    const T* begin() const { return data_; }
    const T* end() const { return data_ + N; }
};

void range_for_test_no_warn() {
    Array<Item, 3> arr = {};
    for (const auto& item : arr) {
        (void)item.value;
    }
}

void range_for_test_c_array() {
    Item items[4] = {};
    for (const auto& item : items) {
        (void)item.value;
    }
}

void range_for_test_deref_still_warns(int* _Nullable p) {
    (void)*p; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}
