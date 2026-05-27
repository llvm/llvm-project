// Tests for the all-returns-nonnull inference. When every return path in a
// function returns a provably non-null expression, the analysis infers the
// function's return as implicitly _Nonnull. Callers within the same TU then
// narrow the returned pointer, suppressing false-positive nullable warnings.
//
// RUN: %clang_cc1 -fsyntax-only -fflow-sensitive-nullability -fnullability-default=nullable -Wno-nullable-to-nonnull-conversion -std=c++17 -Rnullsafe-evidence %s -verify

// ===----------------------------------------------------------------------===//
// Common types
// ===----------------------------------------------------------------------===//

struct Node {
    int value;
    Node * _Nullable next;
};

struct Widget {
    int x;
    int y;
    Node node;
    int data[4];
};

// ===----------------------------------------------------------------------===//
// Provably nonnull return patterns — each should trigger the summary remark
// ===----------------------------------------------------------------------===//

// --- Pattern 1: return address-of member (free function) ---
Node *getNode(Widget *_Nonnull w) { // expected-remark{{function 'getNode' always returns a non-null pointer}}
    return &w->node; // expected-remark-re{{function 'getNode' of global scope (declared at {{.*}}) returns nonnull}}
}

// --- Pattern 2: return this ---
struct Self {
    int val;
    Self *getSelf() { // expected-remark{{function 'getSelf' always returns a non-null pointer}}
        return this; // expected-remark-re{{function 'getSelf' of Self (declared at {{.*}}) returns nonnull}}
    }
};

// --- Pattern 3: return new ---
Node *makeNode() { // expected-remark{{function 'makeNode' always returns a non-null pointer}}
    return new Node(); // expected-remark-re{{function 'makeNode' of global scope (declared at {{.*}}) returns nonnull}}
}

// --- Pattern 4: return address-of local ---
int *getLocal() { // expected-remark{{function 'getLocal' always returns a non-null pointer}}
    static int storage = 42;
    return &storage; // expected-remark-re{{function 'getLocal' of global scope (declared at {{.*}}) returns nonnull}}
}

// --- Pattern 5: return static_cast<T*>(this) ---
struct Base {
    int v;
};
struct Derived : Base {
    Base *asBase() { // expected-remark{{function 'asBase' always returns a non-null pointer}}
        return static_cast<Base *>(this); // expected-remark-re{{function 'asBase' of Derived (declared at {{.*}}) returns nonnull}}
    }
};

// ===----------------------------------------------------------------------===//
// Multi-return: all paths nonnull
// ===----------------------------------------------------------------------===//

Node *getNodeOrNew(Widget *_Nonnull w, bool flag) { // expected-remark{{function 'getNodeOrNew' always returns a non-null pointer}}
    if (flag)
        return &w->node; // expected-remark{{returns nonnull}}
    return new Node(); // expected-remark{{returns nonnull}}
}

// ===----------------------------------------------------------------------===//
// Multi-return: NOT all paths nonnull (no summary remark expected)
// ===----------------------------------------------------------------------===//

Node *getNodeOrNull(Widget *_Nonnull w, bool flag) {
    if (flag)
        return &w->node; // expected-remark{{returns nonnull}}
    return nullptr; // expected-remark{{returns nullable}}
}

// ===----------------------------------------------------------------------===//
// Void return / non-pointer return — should NOT trigger
// ===----------------------------------------------------------------------===//

void doNothing() {}
int getInt() { return 42; }

// ===----------------------------------------------------------------------===//
// Caller-side narrowing: calling a proven all-returns-nonnull function
// should NOT produce a nullable dereference warning
// ===----------------------------------------------------------------------===//

// Free function call via variable init
void caller_via_var(Widget *_Nonnull w) {
    Node *n = getNode(w); // expected-remark-re{{parameter 'w' of 'getNode' (declared at {{.*}}) called with nonnull argument}}
    n->value = 1; // OK - getNode always returns nonnull
}

// Free function call, direct arrow deref (no intermediate variable)
void caller_direct_arrow(Widget *_Nonnull w) {
    getNode(w)->value = 1; // OK - getNode always returns nonnull // expected-remark-re{{parameter 'w' of 'getNode' (declared at {{.*}}) called with nonnull argument}}
}

// Method returning this
void caller_this_ptr() {
    Self s;
    Self *p = s.getSelf();
    p->val = 1; // OK - getSelf always returns nonnull
}

// New expression
void caller_new() {
    Node *n = makeNode();
    n->value = 1; // OK - makeNode always returns nonnull
}

// Multi-return, all nonnull
void caller_multi_return(Widget *_Nonnull w) {
    Node *n = getNodeOrNew(w, true); // expected-remark-re{{parameter 'w' of 'getNodeOrNew' (declared at {{.*}}) called with nonnull argument}}
    n->value = 1; // OK - all returns are nonnull
}

// NOT all-returns-nonnull — should still warn
void caller_nullable_still_warns(Widget *_Nonnull w) {
    Node *n = getNodeOrNull(w, true); // expected-remark-re{{parameter 'w' of 'getNodeOrNull' (declared at {{.*}}) called with nonnull argument}}
    n->value = 1; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

// ===----------------------------------------------------------------------===//
// Getter pattern: the main motivating use case. A class with multiple
// getters returning &member_ should all be inferred as nonnull.
// ===----------------------------------------------------------------------===//

struct Config {
    Node node_a;
    Node node_b;
    Node node_c;
    Widget widget;
    int flags;

    Node *getA() { // expected-remark{{function 'getA' always returns a non-null pointer}}
        return &node_a; // expected-remark{{returns nonnull}}
    }
    Node *getB() { // expected-remark{{function 'getB' always returns a non-null pointer}}
        return &node_b; // expected-remark{{returns nonnull}}
    }
    Node *getC() { // expected-remark{{function 'getC' always returns a non-null pointer}}
        return &node_c; // expected-remark{{returns nonnull}}
    }
    Widget *getWidget() { // expected-remark{{function 'getWidget' always returns a non-null pointer}}
        return &widget; // expected-remark{{returns nonnull}}
    }
    int *getFlags() { // expected-remark{{function 'getFlags' always returns a non-null pointer}}
        return &flags; // expected-remark{{returns nonnull}}
    }
};

void use_config_getters(Config *_Nonnull cfg) {
    // All of these should be nonnull — no warnings.
    cfg->getA()->value = 1;     // OK
    cfg->getB()->value = 2;     // OK
    cfg->getC()->value = 3;     // OK
    cfg->getWidget()->x = 10;   // OK
    *cfg->getFlags() = 0xFF;    // OK
}

// Via variables too
void use_config_via_vars(Config *_Nonnull cfg) {
    Node *a = cfg->getA();
    Node *b = cfg->getB();
    Widget *w = cfg->getWidget();
    int *f = cfg->getFlags();
    a->value = 1;   // OK
    b->value = 2;   // OK
    w->x = 10;      // OK
    *f = 0xFF;       // OK
}

// ===----------------------------------------------------------------------===//
// Edge cases
// ===----------------------------------------------------------------------===//

// Redundant annotation — inference + annotation should not clash
Node *_Nonnull getAlreadyAnnotated(Widget *_Nonnull w) { // expected-remark{{function 'getAlreadyAnnotated' always returns a non-null pointer}}
    return &w->node; // expected-remark{{returns nonnull}}
}

// Narrowed variable: all paths return non-null via narrowing
Node *getNarrowed(Node *p) { // expected-remark{{function 'getNarrowed' always returns a non-null pointer}}
    if (!p)
        return new Node(); // expected-remark{{returns nonnull}}
    return p; // expected-remark{{returns nonnull}}
}

void caller_narrowed() {
    Node *n = getNarrowed(nullptr); // expected-remark-re{{parameter 'p' of 'getNarrowed' (declared at {{.*}}) called with nullable argument}}
    n->value = 1; // OK - getNarrowed always returns nonnull
}

// Direct deref of new expression (not via all-returns-nonnull, but a
// non-null expression deref should not warn either)
void direct_new_deref() {
    (new Node())->value = 1; // OK - new never returns null
}
