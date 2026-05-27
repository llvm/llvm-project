// Consolidated tests for C++-specific features with flow-sensitive nullability.
// Covers: templates, lambdas, coroutines, structured bindings, smart pointers,
// conversion operators, new expressions, exceptions, if-constexpr, and
// nullable-default template return types.
//
// RUN: %clang_cc1 -fsyntax-only -fflow-sensitive-nullability -fnullability-default=nullable -std=c++20 -fcxx-exceptions -I%S/Inputs %s -verify

// ===----------------------------------------------------------------------===//
// Shared type definitions
// ===----------------------------------------------------------------------===//

struct Node {
    int value;
    Node * _Nullable next;
};

Node * _Nullable getNode();
Node * _Nonnull getSafeNode();

// ===----------------------------------------------------------------------===//
// Templates
// ===----------------------------------------------------------------------===//

#pragma clang assume_nonnull begin

template <typename T>
void template_deref_unchecked(T * _Nullable p) {
    (void)p->value; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

template <typename T>
void template_deref_guarded(T * _Nullable p) {
    if (p)
        (void)p->value; // OK -- narrowed
}

template <typename T>
void template_deref_nonnull(T * _Nonnull p) {
    (void)p->value; // OK -- _Nonnull
}

void template_test_functions() {
    Node * _Nullable n = getNode();
    template_deref_unchecked(n);
    template_deref_guarded(n);
    template_deref_nonnull(getSafeNode());
}

// === Template class with nullable member ===

template <typename T>
struct Wrapper {
    T * _Nullable ptr;

    void use_unchecked() {
        (void)ptr->value; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
    }

    void use_guarded() {
        if (ptr)
            (void)ptr->value; // OK
    }
};

void template_test_class() {
    Wrapper<Node> w;
    w.use_unchecked();
    w.use_guarded();
}

// === Template with multiple pointer params of different nullability ===

template <typename T>
void template_mixed_nullability(T * _Nonnull safe, T * _Nullable risky) {
    (void)safe->value; // OK -- _Nonnull
    (void)risky->value; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

void template_test_mixed() {
    template_mixed_nullability(getSafeNode(), getNode());
}

// === Template that narrows then uses ===

template <typename T>
T* _Nullable template_find(T * _Nullable head, int target) {
    for (T * _Nullable p = head; p; p = p->next) {
        if (p->value == target) // OK -- narrowed by loop condition
            return p;
    }
    return nullptr;
}

void template_test_find() {
    Node * _Nullable head = getNode();
    template_find(head, 42);
}

// === Template with cast -- the key false-positive scenario ===
// Template instantiation can produce casts with _Nullable in the dest type.
// The analysis should look through these casts to the source type.

template <typename T>
T* template_cast_and_use(void *raw) {
    T *p = static_cast<T *>(raw);
    // raw is void* (unannotated in nullable-default mode), but static_cast
    // may bake the template param's nullability into the result.
    // Should not warn -- source (raw) is not explicitly _Nullable.
    (void)p->value; // OK -- unannotated source through cast
    return p;
}

void template_test_cast() {
    int dummy;
    template_cast_and_use<Node>(&dummy);
}

// === Non-type template parameters (no effect on nullability) ===

template <int N>
void template_fixed_iteration(Node * _Nullable p) {
    if (!p) return;
    for (int i = 0; i < N; i++)
        (void)p->value; // OK -- narrowed
}

void template_test_non_type() {
    template_fixed_iteration<10>(getNode());
}

// === Template with auto return type ===

template <typename T>
auto template_safe_access(T * _Nullable p, int fallback) {
    if (p)
        return p->value; // OK
    return fallback;
}

void template_test_auto_return() {
    template_safe_access(getNode(), -1);
}

// === Dependent type that resolves to pointer ===

template <typename T>
struct PointerHolder {
    using Ptr = T*;
    Ptr _Nullable held;

    void use() {
        if (held)
            (void)held->value; // OK -- narrowed
    }
};

void template_test_dependent_type() {
    PointerHolder<Node> h;
    h.use();
}

#pragma clang assume_nonnull end

// ===----------------------------------------------------------------------===//
// Lambdas
// ===----------------------------------------------------------------------===//

#pragma clang assume_nonnull begin

// === Capture nullable by value -- warns inside lambda ===

void lambda_test_capture_nullable_by_value(Node * _Nullable p) {
    auto f = [p]() {
        (void)p->value; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
    };
    f();
}

// === Capture narrowed by value -- still nullable inside lambda ===
// Even though p was narrowed before the lambda, the capture creates a new
// copy. The analysis treats each function body independently.

void lambda_test_capture_narrowed_by_value(Node * _Nullable p) {
    if (p) {
        auto f = [p]() {
            // p is captured by value from narrowed context, but the lambda
            // is a separate function body. The analysis sees p as the
            // lambda's parameter (implicitly nullable in nullable-default).
            (void)p->value; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
        };
        f();
        (void)p->value; // OK -- still narrowed in outer scope
    }
}

// === Capture by reference -- narrowing does not propagate ===

void lambda_test_capture_by_ref(Node * _Nullable p) {
    if (p) {
        auto f = [&p]() {
            (void)p->value; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
        };
        f();
    }
}

// === Lambda with its own null check ===

void lambda_test_own_check(Node * _Nullable p) {
    auto f = [p]() {
        if (p)
            (void)p->value; // OK -- narrowed inside lambda
    };
    f();
}

// === Immediately-invoked lambda expression ===

void lambda_test_iife(Node * _Nullable p) {
    [p]() {
        if (p)
            (void)p->value; // OK -- narrowed
    }();
}

// === Lambda capturing nonnull pointer ===

void lambda_test_capture_nonnull(Node * _Nonnull p) {
    auto f = [p]() {
        (void)p->value; // OK -- _Nonnull captured
    };
    f();
}

// === Generic lambda with auto parameter ===

void lambda_test_generic() {
    auto f = [](auto * _Nullable p) {
        if (p)
            (void)p->value; // OK -- narrowed
    };
    Node * _Nullable n = nullptr;
    f(n);
}

// === Lambda returning nullable pointer ===

void lambda_test_return() {
    Node * _Nullable n = nullptr;
    auto getter = [&n]() -> Node * _Nullable { return n; };
    Node * _Nullable result = getter();
    (void)result->value; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

// === Nested lambdas ===

void lambda_test_nested(Node * _Nullable p) {
    auto outer = [p]() {
        auto inner = [p]() {
            (void)p->value; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
        };
        inner();
    };
    outer();
}

// === Lambda with no captures -- unrelated pointer ===

void lambda_test_no_capture() {
    auto f = [](Node * _Nullable p) {
        if (!p) return;
        (void)p->value; // OK -- narrowed by early return
    };
    f(nullptr);
}

// === Mutable lambda modifying captured pointer ===

void lambda_test_mutable_capture(Node * _Nullable p) {
    auto f = [p]() mutable {
        p = nullptr; // mutate the captured copy
        (void)p->value; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
    };
    f();
}

// === Init-capture (C++14) -- captures are independent variables ===

void lambda_test_init_capture_warns() {
    auto f = [p = getNode()]() {
        (void)p->value; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
    };
    f();
}

void lambda_test_init_capture_with_check() {
    auto f = [p = getNode()]() {
        if (p)
            (void)p->value; // OK -- narrowed inside lambda
    };
    f();
}

#pragma clang assume_nonnull end

// === Lambda returning unannotated pointer (regression: getName() crash) ===
// Lambdas have non-identifier names (operator()). The return evidence handler
// must not call getName() on them. This test crashes the compiler if the guard
// is missing (assertion: "Name is not a simple identifier").
// These tests are outside assume_nonnull so unannotated pointers are nullable.

void lambda_test_return_unannotated() {
    Node * _Nullable n = nullptr;
    // Unannotated return type — triggers return evidence emission.
    auto getter = [&n]() -> Node * { return n; };
    auto maker = [&n]() { return n; };  // deduced return type
    Node *r1 = getter();
    (void)r1->value; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
    Node *r2 = maker();
    (void)r2->value; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

// Nested lambda returning pointer — double nesting shouldn't crash.
void lambda_test_return_nested_unannotated() {
    auto outer = []() -> Node * {
        auto inner = []() -> Node * { return nullptr; };
        return inner();
    };
    Node *n = outer();
    (void)n->value; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

// ===----------------------------------------------------------------------===//
// Coroutines
// ===----------------------------------------------------------------------===//

#include "std-coroutine.h"

// --- Generator coroutine type ---

struct Generator {
    struct promise_type {
        Node * _Nullable current;
        Generator get_return_object() { return {}; }
        std::suspend_always initial_suspend() { return {}; }
        std::suspend_always final_suspend() noexcept { return {}; }
        void unhandled_exception() {}
        std::suspend_always yield_value(Node * _Nullable val) {
            current = val;
            return {};
        }
        void return_void() {}
    };
};

// --- Task coroutine type ---

struct Task {
    struct promise_type {
        Task get_return_object() { return {}; }
        std::suspend_never initial_suspend() { return {}; }
        std::suspend_always final_suspend() noexcept { return {}; }
        void unhandled_exception() {}
        void return_void() {}
    };
};

// --- Awaitable that returns a nullable pointer ---

struct NullableAwaitable {
    bool await_ready() const noexcept { return true; }
    void await_suspend(std::coroutine_handle<>) const noexcept {}
    Node * _Nullable await_resume() const noexcept { return nullptr; }
};

#pragma clang assume_nonnull begin

// === Basic coroutine with nullable check ===

Generator coroutine_yield_nodes(Node * _Nullable head) {
    for (Node * _Nullable p = head; p; p = p->next) {
        (void)p->value; // OK -- narrowed by loop condition
        co_yield p;
    }
}

// === co_await returning nullable ===

Task coroutine_consume_awaitable() {
    NullableAwaitable awaitable;
    Node * _Nullable result = co_await awaitable;
    if (result) {
        (void)result->value; // OK -- narrowed
    }
    co_return;
}

// === Null check before co_yield ===

Generator coroutine_guarded_yield(Node * _Nullable n) {
    if (n) {
        (void)n->value; // OK -- narrowed
        co_yield n;
        (void)n->value; // OK -- still narrowed (no reassignment)
    }
}

// === Multiple co_yields with independent checks ===

Generator coroutine_multi_yield(Node * _Nullable a, Node * _Nullable b) {
    if (a) {
        co_yield a;
    }
    if (b) {
        co_yield b;
    }
}

// === Coroutine with nonnull parameter ===

Generator coroutine_nonnull_param(Node * _Nonnull n) {
    (void)n->value; // OK -- _Nonnull
    co_yield n;
    (void)n->value; // OK -- _Nonnull
}

// === Unchecked nullable deref in coroutine body -- should warn ===

Task coroutine_test_unchecked_deref(Node * _Nullable p) {
    (void)p->value; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
    co_return;
}

// === co_await result used without check -- should warn ===

Task coroutine_test_unchecked_co_await_result() {
    NullableAwaitable awaitable;
    Node * _Nullable result = co_await awaitable;
    (void)result->value; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

#pragma clang assume_nonnull end

// ===----------------------------------------------------------------------===//
// Structured Bindings
// ===----------------------------------------------------------------------===//

// Pair-like type for structured bindings
struct PtrPair {
    Node * _Nullable first;
    Node * _Nullable second;
};

PtrPair getPair();

// Tuple-like for testing get<> protocol
struct Triple {
    Node * _Nullable a;
    Node * _Nullable b;
    int c;
};

Triple getTriple();

#pragma clang assume_nonnull begin

// === Basic struct decomposition ===
// Structured binding variables are BindingDecls, not VarDecls.
// The analysis does not currently track narrowing on BindingDecls,
// so these accesses do not warn even without null checks.
// This is a known false negative -- documenting that no crash occurs.

void binding_test_struct_decomp() {
    PtrPair pair = getPair();
    auto [p, q] = pair;
    if (p) {
        (void)p->value; // OK -- narrowed (even though binding)
    }
    if (q) {
        (void)q->value; // OK
    }
}

// === Decomposition with && guard ===

void binding_test_decomp_both_checked() {
    auto [p, q] = getPair();
    if (p && q) {
        (void)p->value; // OK
        (void)q->value; // OK
    }
}

// === Mixed nullable/non-nullable struct ===

struct MixedPair {
    Node * _Nonnull safe;
    Node * _Nullable risky;
};

MixedPair getMixed();

void binding_test_mixed_decomp() {
    auto [safe, risky] = getMixed();
    (void)safe->value;  // OK -- source is _Nonnull
}

void binding_test_mixed_decomp_guarded() {
    auto [safe, risky] = getMixed();
    (void)safe->value; // OK
    if (risky) {
        (void)risky->value; // OK -- checked
    }
}

// === Decomposition from triple ===

void binding_test_triple_decomp() {
    auto [a, b, c] = getTriple();
    if (a && b) {
        (void)a->value; // OK
        (void)b->value; // OK
    }
    (void)c; // OK -- int, not a pointer
}

// === Reference binding through structured bindings ===

void binding_test_ref_decomp() {
    PtrPair pair = getPair();
    auto &[p, q] = pair;
    if (p) {
        (void)p->value; // OK
    }
}

// === Workaround: capture into local variable for narrowing ===

void binding_test_capture_workaround() {
    auto [first, second] = getPair();
    Node * _Nullable p = first;
    Node * _Nullable q = second;
    if (p && q) {
        (void)p->value; // OK -- local VarDecl is tracked
        (void)q->value; // OK
    }
}

// === Structured binding in if-init (C++17) ===

void binding_test_if_init_decomp() {
    if (auto [p, q] = getPair(); p && q) {
        (void)p->value; // OK
        (void)q->value; // OK
    }
}

// === Structured binding in for-range-init ===

struct PairList {
    PtrPair pairs[3];
    PtrPair *begin() { return pairs; }
    PtrPair *end() { return pairs + 3; }
};

void binding_test_range_decomp(PairList &list) {
    for (auto [p, q] : list) {
        if (p) {
            (void)p->value; // OK
        }
    }
}

// === Decomposition of stack-allocated struct ===

void binding_test_stack_decomp() {
    int x = 42;
    Node node{0, nullptr};
    struct { Node * _Nonnull p; int *q; } s = {&node, &x};
    auto [p, q] = s;
    (void)p->value; // OK -- source is _Nonnull
}

#pragma clang assume_nonnull end

// ===----------------------------------------------------------------------===//
// Smart Pointers
// ===----------------------------------------------------------------------===//

// Minimal std smart pointer mocks -- must be in namespace std for detection.
namespace std {

template <typename T>
struct unique_ptr {
    T* ptr;
    using pointer = T*;
    using element_type = T;
    pointer operator->() const { return ptr; }
    element_type& operator*() const { return *ptr; }
    pointer _Nullable get() const { return ptr; }
    explicit operator bool() const { return ptr != nullptr; }
    void reset() { ptr = nullptr; }
    void reset(T* p) { ptr = p; }
    explicit unique_ptr(T* p) : ptr(p) {}
    unique_ptr() : ptr(nullptr) {}
    unique_ptr(unique_ptr&& other) : ptr(other.ptr) { other.ptr = nullptr; }
    unique_ptr& operator=(unique_ptr&& other) { ptr = other.ptr; other.ptr = nullptr; return *this; }
    unique_ptr(const unique_ptr&) = delete;
    unique_ptr& operator=(const unique_ptr&) = delete;
    friend bool operator!=(const unique_ptr& a, decltype(nullptr)) { return a.ptr != nullptr; }
    friend bool operator==(const unique_ptr& a, decltype(nullptr)) { return a.ptr == nullptr; }
    friend bool operator!=(decltype(nullptr), const unique_ptr& a) { return a.ptr != nullptr; }
    friend bool operator==(decltype(nullptr), const unique_ptr& a) { return a.ptr == nullptr; }
};

template <typename T>
struct shared_ptr {
    T* ptr;
    T* operator->() { return ptr; }
    T& operator*() { return *ptr; }
    T* _Nullable get() { return ptr; }
    explicit operator bool() const { return ptr != nullptr; }
    void reset() { ptr = nullptr; }
    void reset(T* p) { ptr = p; }
    friend bool operator!=(const shared_ptr& a, decltype(nullptr)) { return a.ptr != nullptr; }
    friend bool operator==(const shared_ptr& a, decltype(nullptr)) { return a.ptr == nullptr; }
    friend bool operator!=(decltype(nullptr), const shared_ptr& a) { return a.ptr != nullptr; }
    friend bool operator==(decltype(nullptr), const shared_ptr& a) { return a.ptr == nullptr; }
};

template <typename T, typename... Args>
unique_ptr<T> make_unique(Args&&... args);

template <typename T, typename... Args>
shared_ptr<T> make_shared(Args&&... args);

template <typename T>
T&& move(T& t) noexcept;

} // namespace std

#pragma clang assume_nonnull begin

// Non-std smart pointer (should NOT trigger smart pointer warnings)
template <typename T>
struct CustomPtr {
    T* ptr;
    T* operator->() { return ptr; }
    T& operator*() { return *ptr; }
};

// Iterator (should NOT trigger smart pointer warnings)
struct Container {
    struct Iterator {
        Node* ptr;
        Node* operator->() { return ptr; }
        Node& operator*() { return *ptr; }
    };
    Iterator begin();
    Iterator end();
};

// Minimal list mock for container _Nonnull element propagation tests
namespace std {
template <typename T>
struct list {
    struct const_iterator {
        const T* _ptr;
        const T& operator*() const { return *_ptr; }
        const_iterator& operator++() { return *this; }
        bool operator!=(const const_iterator& o) const { return _ptr != o._ptr; }
    };
    const_iterator begin() const;
    const_iterator end() const;
};
} // namespace std

// --- Basic dereference warnings ---

void smartptr_test_deref_warns(std::unique_ptr<Node> sp) {
    sp->value = 1; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

void smartptr_test_shared_deref_warns(std::shared_ptr<Node> sp) {
    sp->value = 1; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

// --- Narrowing via null check ---

void smartptr_test_narrowed_by_check(std::unique_ptr<Node> sp) {
    if (sp) {
        sp->value = 1; // OK -- narrowed by bool check
    }
}

void smartptr_test_narrowed_negated(std::unique_ptr<Node> sp) {
    if (!sp)
        return;
    sp->value = 1; // OK -- narrowed by early return
}

// --- Narrowing via operator!= / operator== with nullptr ---

void smartptr_test_narrowed_ne_nullptr(std::unique_ptr<Node> sp) {
    if (sp != nullptr) {
        sp->value = 1; // OK -- narrowed by != nullptr
    }
}

void smartptr_test_narrowed_ne_nullptr_shared(std::shared_ptr<Node> sp) {
    if (sp != nullptr) {
        sp->value = 1; // OK -- narrowed by != nullptr
    }
}

void smartptr_test_narrowed_ne_nullptr_reversed(std::unique_ptr<Node> sp) {
    if (nullptr != sp) {
        sp->value = 1; // OK -- narrowed by nullptr != sp
    }
}

void smartptr_test_narrowed_eq_nullptr_early_return(std::unique_ptr<Node> sp) {
    if (sp == nullptr)
        return;
    sp->value = 1; // OK -- narrowed by == nullptr early return
}

void smartptr_test_narrowed_ne_nullptr_negated(std::unique_ptr<Node> sp) {
    if (!(sp != nullptr))
        return;
    sp->value = 1; // OK -- !(sp != nullptr) means null, so false edge is narrowed
}

void smartptr_test_not_narrowed_outside_check(std::unique_ptr<Node> sp) {
    if (sp != nullptr) {
        sp->value = 1; // OK
    }
    sp->value = 1; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

// --- Short-circuit || with smart pointer null checks ---

void smartptr_test_or_short_circuit(std::unique_ptr<Node> sp, int* _Nullable other) {
    // anchor == nullptr || anchor->method() — RHS only runs when sp is non-null
    if (sp == nullptr || sp->value == 0) {
        return;
    }
    sp->value = 1; // OK -- sp narrowed
}

void smartptr_test_or_short_circuit_rhs_deref(std::unique_ptr<Node> sp) {
    // The RHS of || only evaluates when LHS is false (sp != nullptr)
    if (sp == nullptr || sp->value == 42) { // OK -- sp narrowed on RHS of ||
        return;
    }
}

void smartptr_test_or_short_circuit_raw_ptr(int* _Nullable p) {
    // Same pattern with raw pointer for comparison
    if (p == nullptr || *p == 42) { // OK -- p narrowed on RHS of ||
        return;
    }
}

// --- make_unique/make_shared narrow ---

void smartptr_test_make_unique_narrows() {
    auto sp = std::make_unique<Node>();
    sp->value = 1; // OK -- make_unique always returns non-null
}

void smartptr_test_make_shared_narrows() {
    auto sp = std::make_shared<Node>();
    sp->value = 1; // OK -- make_shared always returns non-null
}

// --- new-expression narrows (throwing new never returns null) ---

void smartptr_test_unique_ptr_new_narrows() {
    std::unique_ptr<Node> sp(new Node());
    sp->value = 1; // OK -- new Node() never returns null (throwing new)
}

void smartptr_test_assign_from_new() {
    std::unique_ptr<Node> sp;
    sp->value = 1; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
    sp = std::unique_ptr<Node>(new Node());
    sp->value = 1; // OK -- reassignment from new-expression narrows
}

// --- reset() makes nullable ---

void smartptr_test_reset_makes_nullable(std::unique_ptr<Node> sp) {
    if (sp) {
        sp->value = 1; // OK
    }
    sp.reset();
    sp->value = 1; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

void smartptr_test_reset_with_arg_narrows(std::unique_ptr<Node> sp) {
    sp.reset(new Node());
    sp->value = 1; // OK -- reset(ptr) gives it a value
}

void smartptr_test_reset_nullptr_stays_nullable(std::unique_ptr<Node> sp) {
    if (sp) {
        sp->value = 1; // OK -- narrowed
    }
    sp.reset(nullptr);
    sp->value = 1; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

// --- std::move makes source nullable ---

void smartptr_test_move_makes_source_nullable(std::unique_ptr<Node> sp) {
    if (sp) {
        sp->value = 1; // OK
    }
    auto other = std::move(sp);
    sp->value = 1; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

// --- Member smart pointers ---

struct Owner {
    std::unique_ptr<Node> csm_;

    void use_no_evidence() {
        csm_->value = 1; // OK -- no evidence of nullability
    }

    void use_after_reset() {
        csm_->value = 1; // OK -- before reset
        csm_.reset();
        csm_->value = 1; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
    }

    void use_after_reset_with_arg() {
        csm_.reset(new Node());
        csm_->value = 1; // OK -- reset(ptr) narrows
    }

    void use_after_move() {
        auto other = std::move(csm_);
        csm_->value = 1; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
    }

    void use_safe_after_reset() {
        csm_.reset();
        if (csm_) {
            csm_->value = 1; // OK -- narrowed
        }
    }

    void use_get_after_reset_nonnull() {
        csm_.reset(new Node());
        csm_.get()->value = 1; // OK -- narrowed via reset(nonnull)
    }

    void use_get_after_reset_null() {
        csm_.reset();
        csm_.get()->value = 1; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
    }
};

// --- Assignment from make_unique re-narrows ---

void smartptr_test_assign_from_make_unique() {
    std::unique_ptr<Node> sp;
    sp->value = 1; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
    sp = std::make_unique<Node>();
    sp->value = 1; // OK -- assignment from make_unique narrows
}

// --- Move-construct (VarDecl init) inherits source's narrowed state ---
// Real libc++'s `auto dst = std::move(src);` wraps std::move in a
// CXXConstructExpr (the unique_ptr move ctor). Earlier the standalone
// std::move handler erased the source before the DeclStmt could read
// its pre-move state, leaving the move target falsely nullable.

void smartptr_test_move_construct_target_narrowed() {
    auto owner = std::make_unique<Node>();
    auto new_owner = std::move(owner);
    new_owner->value = 1;  // OK -- received ownership from narrowed source
    owner->value = 1;      // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

// --- Non-std smart pointers should NOT warn ---

void smartptr_test_custom_ptr_no_warn(CustomPtr<Node> cp) {
    cp->value = 1; // OK -- not a std smart pointer, skip operator->
}

void smartptr_test_iterator_no_warn(Container c) {
    auto it = c.begin();
    it->value = 1; // OK -- iterator, not a smart pointer
}

// --- Container _Nonnull element propagation through iterators ---
// _Nonnull on a smart pointer inside a container's template argument should
// propagate through iterator dereference to suppress false positive warnings.

class ContainerNonnullTest {
    std::list<std::unique_ptr<Node> _Nonnull> nonnull_list_;
    std::list<std::unique_ptr<Node> _Nullable> nullable_list_;
    std::list<std::unique_ptr<Node>> plain_list_;

    void nonnull_range_for_no_warn() {
        for (const auto& entry : nonnull_list_) {
            entry->value = 1; // no warning — _Nonnull element type
        }
    }

    void nullable_range_for_warns() {
        for (const auto& entry : nullable_list_) {
            entry->value = 1; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
        }
    }

    void plain_range_for_warns() {
        for (const auto& entry : plain_list_) {
            entry->value = 1; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
        }
    }
};

// --- .get() on un-narrowed smart pointer warns ---

void smartptr_test_get_warns_unnarrowed(std::unique_ptr<Node> sp) {
    sp.get()->value = 1; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

// --- .get() propagates smart pointer narrowing state ---

void smartptr_test_get_propagates_nonnull() {
    auto sp = std::make_unique<Node>();
    Node* raw = sp.get();
    raw->value = 1; // no warning — sp is nonnull, get() propagates that
}

void smartptr_test_get_propagates_nullable() {
    std::unique_ptr<Node> empty;
    Node* raw = empty.get();
    raw->value = 1; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

// --- .get() direct dereference respects narrowing ---

void smartptr_test_get_direct_deref_narrowed() {
    auto sp = std::make_unique<Node>();
    sp.get()->value = 1; // no warning — sp is narrowed via make_unique
}

void smartptr_test_get_direct_deref_after_check(std::unique_ptr<Node> sp) {
    if (sp) {
        sp.get()->value = 1; // no warning — sp is narrowed via null check
    }
}

void smartptr_test_get_direct_deref_after_reset_nonnull(std::unique_ptr<Node> sp) {
    sp.reset(new Node());
    sp.get()->value = 1; // no warning — sp is narrowed via reset(nonnull)
}

void smartptr_test_get_direct_deref_after_reset_null(std::unique_ptr<Node> sp) {
    sp.reset();
    sp.get()->value = 1; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

// --- Raw pointers still work as before ---

void smartptr_test_raw_ptr_still_warns(Node* _Nullable p) {
    p->value = 1; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

void smartptr_test_raw_ptr_narrowed(Node* _Nullable p) {
    if (p) {
        p->value = 1; // OK -- narrowed
    }
}

#pragma clang assume_nonnull end

// ===----------------------------------------------------------------------===//
// Conversion Operators
// ===----------------------------------------------------------------------===//

// NOTE: The original test ran under both -fnullability-default=nonnull and
// -fnullability-default=nullable. This consolidated file uses nullable only.
// The key thing being tested is that conversion operators (operator T*())
// don't trigger spurious nullability-inference warnings.

typedef void* bool_type;

struct ConvertToRawPtr {
    void* data;
    operator void*() const { return data; }
};

struct ConvertToTypedef {
    bool_type data;
    operator bool_type() const { return data; }
};

struct ConvertToNonPointer {
    int value;
    operator int() const { return value; }
};

void convop_test_conversions() {
    ConvertToRawPtr a;
    void* p = a;

    ConvertToTypedef b;
    bool_type q = b;

    ConvertToNonPointer c;
    int n = c;
}

void convop_test_deref_still_warns(int* _Nullable p) {
    (void)*p; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

// ===----------------------------------------------------------------------===//
// New Expressions
// ===----------------------------------------------------------------------===//

typedef __SIZE_TYPE__ size_t;

namespace std {
  struct nothrow_t {};
  extern const nothrow_t nothrow;
}

void *operator new(size_t, const std::nothrow_t &) noexcept;

struct Widget {
    int value;
};

Widget * _Nullable getNullableWidget();

#pragma clang assume_nonnull begin

void newexpr_test_direct_deref() {
    Widget *w = new Widget();
    w->value = 42; // OK - throwing new never returns null
}

void newexpr_test_var_deref() {
    Widget *w = new Widget();
    int v = w->value; // OK - narrowed via new
}

void newexpr_test_nothrow_warns() {
    Widget *w = new (std::nothrow) Widget();
    w->value = 42; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

void newexpr_test_nullable_control() {
    Widget *w = getNullableWidget();
    w->value = 42; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

#pragma clang assume_nonnull end

// ===----------------------------------------------------------------------===//
// Exceptions
// ===----------------------------------------------------------------------===//

#pragma clang assume_nonnull begin

// === Narrowing before try block persists inside ===

void exception_test_narrow_before_try(Node * _Nullable p) {
    if (!p) return;
    try {
        (void)p->value; // OK -- narrowed before try
    } catch (...) {
    }
}

// === throw in null guard narrows ===

void exception_test_throw_guard(Node * _Nullable p) {
    if (!p) throw "null";
    (void)p->value; // OK -- throw terminates null path
}

// === try body with null check ===

void exception_test_null_check_in_try(Node * _Nullable p) {
    try {
        if (!p) throw "null";
        (void)p->value; // OK -- narrowed by throw guard
    } catch (...) {
    }
}

// === catch block should not inherit narrowing from try ===

void exception_test_after_try_catch(Node * _Nullable p) {
    try {
        if (p)
            (void)p->value; // OK -- narrowed
    } catch (...) {
    }
    // After try/catch, p's narrowing depends on merge of try and catch edges.
    // Conservative: should warn.
    (void)p->value; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

// === Narrowing in both try and catch ===

void exception_test_narrow_in_both(Node * _Nullable p) {
    try {
        if (!p) throw "null";
        (void)p->value; // OK
    } catch (...) {
        if (!p) return;
        (void)p->value; // OK -- narrowed in catch too
    }
    // Both try (throw guard) and catch (early return) narrowed p,
    // so the merge point should preserve narrowing.
    (void)p->value; // OK -- narrowed on all paths
}

// === Multiple catch blocks ===

void exception_test_multiple_catch(Node * _Nullable p) {
    if (!p) return;
    try {
        (void)p->value; // OK -- narrowed
    } catch (int) {
    } catch (...) {
    }
}

// === throw expression in ternary ===

void exception_test_throw_ternary(Node * _Nullable p) {
    int v = p ? p->value : throw "null"; // OK -- throw terminates null path
    (void)v;
}

// === Noexcept function -- no exception CFG edges ===

void exception_test_noexcept_narrowing(Node * _Nullable p) noexcept {
    if (!p) return;
    (void)p->value; // OK -- narrowed, no exception edges to worry about
}

#pragma clang assume_nonnull end

// ===----------------------------------------------------------------------===//
// if constexpr
// ===----------------------------------------------------------------------===//

#pragma clang assume_nonnull begin

// Known limitation: the warn_null_init_nonnull check fires during
// declaration processing, before if-constexpr discarding. This means
// _Nonnull p = nullptr in a discarded branch still warns. Suppressing
// this would require tracking discarded-branch state at decl processing
// time, which Clang doesn't expose. In practice, writing explicit
// _Nonnull p = nullptr in a discarded branch is very rare.

void constexpr_test_discarded() {
    if constexpr (false) {
        int * _Nonnull p = nullptr; // expected-warning{{null assigned to a variable of nonnull type}}
    }
}

// Live branch correctly warns
void constexpr_test_live() {
    if constexpr (true) {
        int * _Nonnull p = nullptr; // expected-warning{{null assigned to a variable of nonnull type}} expected-warning{{assigning nullable pointer to nonnull variable}} expected-note{{add a null check before assigning}}
    }
}

// Flow analysis narrowing works in live constexpr branches
void constexpr_test_narrowing(int * _Nullable p) {
    if constexpr (true) {
        if (p) {
            *p = 42; // OK -- narrowed
        }
    }
}

// Dereference in live branch warns correctly
void constexpr_test_deref(int * _Nullable p) {
    if constexpr (true) {
        *p = 42; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
    }
}

// Template with if constexpr -- both instantiations checked
template<bool B>
void constexpr_template_branch() {
    if constexpr (B) {
        int * _Nonnull p = nullptr; // expected-warning 2{{null assigned to a variable of nonnull type}} expected-warning{{assigning nullable pointer to nonnull variable}} expected-note{{add a null check before assigning}}
    }
}

void constexpr_instantiate_both() {
    constexpr_template_branch<false>();
    constexpr_template_branch<true>(); // expected-note{{in instantiation}}
}

#pragma clang assume_nonnull end

// ===----------------------------------------------------------------------===//
// Nullable-Default Template Return Types
// ===----------------------------------------------------------------------===//

// Tests that explicit _Nullable return types on template methods are caught.
// Mimics the getComponent<T>() pattern from Clay ECS.

struct Component {
    int value;
    void setValue(int v) { value = v; }
};

struct Entity {
    template<typename T>
    T* _Nullable getComponent() { return nullptr; }

    Component* _Nullable getFirstComponent() { return nullptr; }
};

// Case 1: Non-template function -> local var -> arrow deref
void nullable_template_test_non_template(Entity* e) {
    Component* c = e->getFirstComponent(); // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
    c->setValue(42);     // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

// Case 2: Template function -> local var -> arrow deref
void nullable_template_test_template(Entity* e) {
    Component* c = e->getComponent<Component>(); // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
    c->setValue(42);     // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

// Case 3: Chained call -> local var -> data member access
void nullable_template_test_data_member(Entity* e) {
    Component* c = e->getComponent<Component>(); // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
    c->value = 1;        // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

// Case 4: With null check -- should NOT warn for c, still warns for e
void nullable_template_test_with_check(Entity* e) {
    Component* c = e->getComponent<Component>(); // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
    if (c != nullptr) {
        c->setValue(42); // OK -- c is narrowed
    }
}
