// Tests that a _Nonnull qualifier on the declared type of a smart pointer
// (unique_ptr / shared_ptr / weak_ptr) suppresses deref warnings without
// needing a runtime narrowing (if-check) in the flow.
//
// Motivation: overload resolution on operator->/operator* strips the
// nullability attribute from the argument expression's type, so checking
// `Obj->getType()->getNullability()` is a no-op. The fix walks back to
// the DeclRefExpr / MemberExpr to consult the declared type.
//
// This test requires system C++ headers.
// UNSUPPORTED: target={{.*-windows.*}}
// REQUIRES: system-darwin || system-linux
// RUN: %clangxx -fsyntax-only -fflow-sensitive-nullability -fnullability-default=nullable -std=c++17 %s -Xclang -verify

#include <memory>

struct Widget {
    void draw() const;
    int value;
};

// --- Parameter annotated _Nonnull: deref via operator-> and operator* is clean.
void param_arrow(std::unique_ptr<Widget> _Nonnull w) {
    w->draw();          // OK — declared _Nonnull
}

void param_star(std::unique_ptr<Widget> _Nonnull w) {
    (*w).draw();        // OK — declared _Nonnull
}

void param_shared(std::shared_ptr<Widget> _Nonnull w) {
    w->draw();          // OK — declared _Nonnull
}

// --- Without _Nonnull, unchecked deref still warns.
void param_unannotated(std::unique_ptr<Widget> w) {
    w->draw();          // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

// --- Field annotated _Nonnull: deref through a MemberExpr is clean.
struct Holder {
    std::unique_ptr<Widget> _Nonnull w;
};

void field_arrow(Holder& h) {
    h.w->draw();        // OK — declared _Nonnull on the field
}
