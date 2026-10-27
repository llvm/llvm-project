// Tests for the all-returns-nonnull inference. When every return path in a
// function returns a provably non-null expression, the analysis infers the
// function's return as implicitly _Nonnull. Callers within the same TU then
// narrow the returned pointer, suppressing false-positive nullable warnings.
//
// RUN: %clang_cc1 -fsyntax-only -fnullability-safety -fnullability-default=nullable -Wno-nullable-to-nonnull-conversion -std=c++17 %s -verify
// RUN: rm -f %t.json
// RUN: %clang_cc1 -fsyntax-only -fnullability-default=nullable -std=c++17 %s \
// RUN:   --ssaf-extract-summaries=NullabilitySafety \
// RUN:   --ssaf-compilation-unit-id=tu --ssaf-tu-summary-file=%t.json
// RUN: %python %S/../Analysis/Scalable/NullabilitySafety/Inputs/decode-summary.py %t.json | FileCheck %s --check-prefix=EVIDENCE

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
Node *getNode(Widget *_Nonnull w) {
    return &w->node;
}

// --- Pattern 2: return this ---
struct Self {
    int val;
    Self *getSelf() {
        return this;
    }
};

// --- Pattern 3: return new ---
Node *makeNode() {
    return new Node();
}

// --- Pattern 4: return address-of local ---
int *getLocal() {
    static int storage = 42;
    return &storage;
}

// --- Pattern 5: return static_cast<T*>(this) ---
struct Base {
    int v;
};
struct Derived : Base {
    Base *asBase() {
        return static_cast<Base *>(this);
    }
};

// ===----------------------------------------------------------------------===//
// Multi-return: all paths nonnull
// ===----------------------------------------------------------------------===//

Node *getNodeOrNew(Widget *_Nonnull w, bool flag) {
    if (flag)
        return &w->node;
    return new Node();
}

// ===----------------------------------------------------------------------===//
// Multi-return: NOT all paths nonnull (no summary remark expected)
// ===----------------------------------------------------------------------===//

Node *getNodeOrNull(Widget *_Nonnull w, bool flag) {
    if (flag)
        return &w->node;
    return nullptr;
}

// An unannotated return is unknown: it emits no cross-TU evidence, but must
// still invalidate the local all-returns-nonnull summary.
Node *getNodeOrUnknown(Widget *_Nonnull w, Node *unknown, bool flag) {
    if (flag)
        return &w->node;
    return unknown;
}

void caller_unknown_still_warns(Widget *_Nonnull w, Node *unknown) {
    Node *n = getNodeOrUnknown(w, unknown, true);
    n->value = 1; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
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
    Node *n = getNode(w);
    n->value = 1; // OK - getNode always returns nonnull
}

// Free function call, direct arrow deref (no intermediate variable)
void caller_direct_arrow(Widget *_Nonnull w) {
    getNode(w)->value = 1; // OK - getNode always returns nonnull
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
    Node *n = getNodeOrNew(w, true);
    n->value = 1; // OK - all returns are nonnull
}

// NOT all-returns-nonnull — should still warn
void caller_nullable_still_warns(Widget *_Nonnull w) {
    Node *n = getNodeOrNull(w, true);
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

    Node *getA() {
        return &node_a;
    }
    Node *getB() {
        return &node_b;
    }
    Node *getC() {
        return &node_c;
    }
    Widget *getWidget() {
        return &widget;
    }
    int *getFlags() {
        return &flags;
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
Node *_Nonnull getAlreadyAnnotated(Widget *_Nonnull w) {
    return &w->node;
}

// Narrowed variable: all paths return non-null via narrowing
Node *getNarrowed(Node *p) {
    if (!p)
        return new Node();
    return p;
}

void caller_narrowed() {
    Node *n = getNarrowed(nullptr);
    n->value = 1; // OK - getNarrowed always returns nonnull
}

// Direct deref of new expression (not via all-returns-nonnull, but a
// non-null expression deref should not warn either)
void direct_new_deref() {
    (new Node())->value = 1; // OK - new never returns null
}

// EVIDENCE-NOT:  {{.}}
// EVIDENCE:      c:@F@caller_direct_arrow#*$@S@Widget# NonnullEvidence c:@F@getNode#*$@S@Widget# param 1
// EVIDENCE-NEXT: c:@F@caller_multi_return#*$@S@Widget# NonnullEvidence c:@F@getNodeOrNew#*$@S@Widget#b# param 1
// EVIDENCE-NEXT: c:@F@caller_narrowed# NullableEvidence c:@F@getNarrowed#*$@S@Node# param 1
// EVIDENCE-NEXT: c:@F@caller_nullable_still_warns#*$@S@Widget# NonnullEvidence c:@F@getNodeOrNull#*$@S@Widget#b# param 1
// EVIDENCE-NEXT: c:@F@caller_unknown_still_warns#*$@S@Widget#*$@S@Node# ConditionalEvidence c:@F@getNodeOrUnknown#*$@S@Widget#*$@S@Node#b# param 2 <- c:@F@caller_unknown_still_warns#*$@S@Widget#*$@S@Node# param 2
// EVIDENCE-NEXT: c:@F@caller_unknown_still_warns#*$@S@Widget#*$@S@Node# NonnullEvidence c:@F@getNodeOrUnknown#*$@S@Widget#*$@S@Node#b# param 1
// EVIDENCE-NEXT: c:@F@caller_via_var#*$@S@Widget# NonnullEvidence c:@F@getNode#*$@S@Widget# param 1
// EVIDENCE-NEXT: c:@F@getAlreadyAnnotated#*$@S@Widget# AllReturnsNonnull c:@F@getAlreadyAnnotated#*$@S@Widget# return
// EVIDENCE-NEXT: c:@F@getAlreadyAnnotated#*$@S@Widget# NonnullEvidence c:@F@getAlreadyAnnotated#*$@S@Widget# return
// EVIDENCE-NEXT: c:@F@getLocal# AllReturnsNonnull c:@F@getLocal# return
// EVIDENCE-NEXT: c:@F@getLocal# NonnullEvidence c:@F@getLocal# return
// EVIDENCE-NEXT: c:@F@getNarrowed#*$@S@Node# AllReturnsNonnull c:@F@getNarrowed#*$@S@Node# return
// EVIDENCE-NEXT: c:@F@getNarrowed#*$@S@Node# MaybeNullEvidence c:@F@getNarrowed#*$@S@Node# param 1
// EVIDENCE-NEXT: c:@F@getNarrowed#*$@S@Node# NonnullEvidence c:@F@getNarrowed#*$@S@Node# return
// EVIDENCE-NEXT: c:@F@getNode#*$@S@Widget# AllReturnsNonnull c:@F@getNode#*$@S@Widget# return
// EVIDENCE-NEXT: c:@F@getNode#*$@S@Widget# NonnullEvidence c:@F@getNode#*$@S@Widget# return
// EVIDENCE-NEXT: c:@F@getNodeOrNew#*$@S@Widget#b# AllReturnsNonnull c:@F@getNodeOrNew#*$@S@Widget#b# return
// EVIDENCE-NEXT: c:@F@getNodeOrNew#*$@S@Widget#b# NonnullEvidence c:@F@getNodeOrNew#*$@S@Widget#b# return
// EVIDENCE-NEXT: c:@F@getNodeOrNull#*$@S@Widget#b# NonnullEvidence c:@F@getNodeOrNull#*$@S@Widget#b# return
// EVIDENCE-NEXT: c:@F@getNodeOrNull#*$@S@Widget#b# NullableEvidence c:@F@getNodeOrNull#*$@S@Widget#b# return
// EVIDENCE-NEXT: c:@F@getNodeOrUnknown#*$@S@Widget#*$@S@Node#b# ConditionalEvidence c:@F@getNodeOrUnknown#*$@S@Widget#*$@S@Node#b# return <- c:@F@getNodeOrUnknown#*$@S@Widget#*$@S@Node#b# param 2
// EVIDENCE-NEXT: c:@F@getNodeOrUnknown#*$@S@Widget#*$@S@Node#b# NonnullEvidence c:@F@getNodeOrUnknown#*$@S@Widget#*$@S@Node#b# return
// EVIDENCE-NEXT: c:@F@makeNode# AllReturnsNonnull c:@F@makeNode# return
// EVIDENCE-NEXT: c:@F@makeNode# NonnullEvidence c:@F@makeNode# return
// EVIDENCE-NEXT: c:@S@Config@F@getA# AllReturnsNonnull c:@S@Config@F@getA# return
// EVIDENCE-NEXT: c:@S@Config@F@getA# NonnullEvidence c:@S@Config@F@getA# return
// EVIDENCE-NEXT: c:@S@Config@F@getB# AllReturnsNonnull c:@S@Config@F@getB# return
// EVIDENCE-NEXT: c:@S@Config@F@getB# NonnullEvidence c:@S@Config@F@getB# return
// EVIDENCE-NEXT: c:@S@Config@F@getC# AllReturnsNonnull c:@S@Config@F@getC# return
// EVIDENCE-NEXT: c:@S@Config@F@getC# NonnullEvidence c:@S@Config@F@getC# return
// EVIDENCE-NEXT: c:@S@Config@F@getFlags# AllReturnsNonnull c:@S@Config@F@getFlags# return
// EVIDENCE-NEXT: c:@S@Config@F@getFlags# NonnullEvidence c:@S@Config@F@getFlags# return
// EVIDENCE-NEXT: c:@S@Config@F@getWidget# AllReturnsNonnull c:@S@Config@F@getWidget# return
// EVIDENCE-NEXT: c:@S@Config@F@getWidget# NonnullEvidence c:@S@Config@F@getWidget# return
// EVIDENCE-NEXT: c:@S@Derived@F@asBase# AllReturnsNonnull c:@S@Derived@F@asBase# return
// EVIDENCE-NEXT: c:@S@Derived@F@asBase# NonnullEvidence c:@S@Derived@F@asBase# return
// EVIDENCE-NEXT: c:@S@Self@F@getSelf# AllReturnsNonnull c:@S@Self@F@getSelf# return
// EVIDENCE-NEXT: c:@S@Self@F@getSelf# NonnullEvidence c:@S@Self@F@getSelf# return
// EVIDENCE-NOT:  {{.}}
