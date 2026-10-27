// Tests that flow-sensitive nullability works with real standard library headers,
// not just mocked std:: types. This catches AST wrapping differences
// (ExprWithCleanups, CXXBindTemporaryExpr) that mocks don't produce.
//
// This test requires system C++ headers, so it runs through the driver
// rather than cc1. It is unsupported on targets without a C++ stdlib.
// UNSUPPORTED: target={{.*-windows.*}}
// REQUIRES: system-darwin || system-linux
// RUN: %clangxx -fsyntax-only -fnullability-safety -fnullability-default=nullable -std=c++17 %s -Xclang -verify

#include <memory>
#include <utility>

struct Node {
    virtual ~Node();
    int value;
};

struct DerivedNode : Node {};
Node * _Nullable maybe_node();

#pragma clang assume_nonnull begin

void make_unique_narrows() {
    auto sp = std::make_unique<Node>();
    sp->value = 1; // OK — make_unique always returns non-null
}

void make_shared_narrows() {
    auto sp = std::make_shared<Node>();
    sp->value = 1; // OK — make_shared always returns non-null
}

void move_makes_nullable() {
    auto sp = std::make_unique<Node>();
    sp->value = 1; // OK
    auto other = std::move(sp);
    other->value = 1; // OK — move target inherited source's narrowed state
    sp->value = 1; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

void declared_nonnull_move_sources_warn() {
    std::unique_ptr<Node> _Nonnull constructed = std::make_unique<Node>();
    auto target = std::move(constructed);
    target->value = 1;
    constructed->value = 1; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}

    std::unique_ptr<Node> _Nonnull assigned = std::make_unique<Node>();
    target = std::move(assigned);
    assigned->value = 1; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}

    std::unique_ptr<Node> _Nonnull bare = std::make_unique<Node>();
    static_cast<void>(std::move(bare));
    bare->value = 1; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

void consume_node(std::unique_ptr<Node> p);

void declared_nonnull_bare_move_arg_warns(std::unique_ptr<Node> _Nonnull sp) {
    consume_node(std::move(sp));
    sp->value = 1; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

// `p.reset()` on a real libc++ unique_ptr calls reset(pointer = pointer()),
// so MCE->getArg(0) is a CXXDefaultArgExpr wrapping a value-init. The
// handler must treat that form as "no user-provided arg" and mark the
// pointer null — not as reset-to-nonnull.
void reset_default_arg_makes_nullable() {
    auto sp = std::make_unique<Node>();
    sp->value = 1; // OK
    sp.reset();
    sp->value = 1; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

void reset_argument_flow(Node * _Nullable p, Node &node) {
    auto sp = std::make_unique<Node>();
    sp.reset(p);
    sp->value = 1; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}

    sp.reset(maybe_node());
    sp->value = 1; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}

    sp.reset(dynamic_cast<DerivedNode *>(&node));
    sp->value = 1; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}

    sp.reset(nullptr);
    sp->value = 1; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}

    sp.reset(new Node());
    sp->value = 1;

    sp.reset(&node);
    sp->value = 1;

    if (p) {
        sp.reset(p);
        sp->value = 1;
    }
}

void declared_nonnull_reset_unknown_uses_contract(Node * _Nullable p,
                                                  Node &node) {
    std::unique_ptr<Node> _Nonnull sp = std::make_unique<Node>();
    sp.reset(p);
    sp->value = 1;

    sp.reset(maybe_node());
    sp->value = 1;

    sp.reset(dynamic_cast<DerivedNode *>(&node));
    sp->value = 1;

    sp.reset(nullptr);
    sp->value = 1; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

struct SmartOwner {
    std::unique_ptr<Node> sp;

    void resetMember(Node * _Nullable p, Node &node) {
        sp = std::make_unique<Node>();
        sp.reset(p);
        sp->value = 1; // Unknown reset falls back to the member's declaration.
        sp.reset(dynamic_cast<DerivedNode *>(&node));
        sp->value = 1;
        sp.reset();
        sp->value = 1; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
        sp.reset(new Node());
        sp->value = 1;
        sp.reset(&node);
        sp->value = 1;
        if (p) {
            sp.reset(p);
            sp->value = 1;
        }
    }
};

void reassign_make_unique_renarrows() {
    std::unique_ptr<Node> sp;
    sp->value = 1; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
    sp = std::make_unique<Node>();
    sp->value = 1; // OK — reassignment from make_unique narrows
}

void new_expression_narrows() {
    std::unique_ptr<Node> sp(new Node());
    sp->value = 1; // OK — new Node() never returns null (throwing new)
}

void reassign_from_new_renarrows() {
    std::unique_ptr<Node> sp;
    sp->value = 1; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
    sp = std::unique_ptr<Node>(new Node());
    sp->value = 1; // OK — reassignment from new-expression narrows
}

// C++20 rewrites `sp != nullptr` into `!(sp == nullptr)` via
// CXXRewrittenBinaryOperator. Verify flow narrowing sees through it.
void ne_nullptr_narrows_unique() {
    std::unique_ptr<Node> sp;
    if (sp != nullptr) {
        sp->value = 1; // OK — narrowed by != nullptr
    }
    sp->value = 1; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

void ne_nullptr_narrows_shared(std::shared_ptr<Node> sp) {
    if (sp != nullptr) {
        sp->value = 1; // OK — narrowed by != nullptr
    }
    // shared_ptr params are not treated as nullable with real headers,
    // so no warning expected here (separate issue from narrowing)
}

void eq_nullptr_early_return_narrows() {
    std::unique_ptr<Node> sp;
    if (sp == nullptr)
        return;
    sp->value = 1; // OK — narrowed by == nullptr early return
}

void ne_nullptr_reversed_narrows() {
    std::unique_ptr<Node> sp;
    if (nullptr != sp) {
        sp->value = 1; // OK — narrowed by nullptr != sp
    }
}

// --- || short-circuit and || false-edge narrowing ---

void or_short_circuit_rhs(std::unique_ptr<Node> sp) {
    if (sp == nullptr || sp->value == 42) { // OK — sp narrowed on RHS of ||
        return;
    }
}

void or_short_circuit_bool(std::unique_ptr<Node> sp) {
    if (!sp || sp->value == 42) { // OK — sp narrowed on RHS of ||
        return;
    }
}

void or_false_edge_narrows(std::unique_ptr<Node> sp) {
    if (sp == nullptr || sp->value == 42) {
        return;
    }
    sp->value = 1; // OK — sp narrowed after || early return
}

// When the || RHS returns a type with a destructor (e.g. unique_ptr),
// the CFG inserts temp-destructor cleanup blocks that merge the ||
// operand paths. decomposeOr at the IfStmt level recovers narrowing.
struct Registry {
    std::unique_ptr<Node> Lookup(int id);
    std::unique_ptr<Node> Create();
};

void or_temp_destructor_false_edge(Registry* reg) {
    std::unique_ptr<Node> anchor = reg->Create();
    // RHS creates a temporary unique_ptr whose destructor merges paths
    if (anchor == nullptr || reg->Lookup(anchor->value) == nullptr) {
        return;
    }
    anchor->value = 1; // OK — narrowed despite temp destructor
    int v = anchor->value; // OK
}

void or_temp_destructor_two_vars(Registry* reg,
                                 std::unique_ptr<Node> a,
                                 std::unique_ptr<Node> b) {
    if (a == nullptr || b == nullptr || reg->Lookup(a->value) == nullptr) {
        return;
    }
    a->value = 1; // OK — both narrowed
    b->value = 2; // OK
}

#pragma clang assume_nonnull end
