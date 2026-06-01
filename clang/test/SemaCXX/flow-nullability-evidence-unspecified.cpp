// Tests that evidence emission distinguishes explicit _Nullable from
// null-unspecified (unannotated) sources. Unannotated pointers defaulted to
// nullable by -fnullability-default=nullable should NOT produce "nullable"
// evidence — only explicitly _Nullable or nullptr should.
//
// RUN: %clang_cc1 -fsyntax-only -fflow-sensitive-nullability -fnullability-default=nullable -Wno-nullable-to-nonnull-conversion -std=c++17 -Rnullsafe-evidence %s -verify

// ===----------------------------------------------------------------------===//
// Parameter evidence: explicit _Nullable argument -> "called with nullable"
// ===----------------------------------------------------------------------===//

void takes_ptr(int *p);

void pass_explicit_nullable(int * _Nullable np) {
    takes_ptr(np); // expected-remark-re{{parameter 'p' of 'takes_ptr' (declared at {{.*}}) called with nullable argument}}
}

// ===----------------------------------------------------------------------===//
// Parameter evidence: nullptr -> "called with nullable"
// ===----------------------------------------------------------------------===//

void pass_nullptr() {
    takes_ptr(nullptr); // expected-remark-re{{parameter 'p' of 'takes_ptr' (declared at {{.*}}) called with nullable argument}}
}

// ===----------------------------------------------------------------------===//
// Parameter evidence: unannotated pointer -> NO evidence emitted
// ===----------------------------------------------------------------------===//

void pass_unannotated(int *p) {
    takes_ptr(p); // no remark — p is null-unspecified, not explicitly _Nullable
}

// ===----------------------------------------------------------------------===//
// Parameter evidence: _Nonnull argument -> "called with nonnull" (unchanged)
// ===----------------------------------------------------------------------===//

void pass_nonnull(int * _Nonnull p) {
    takes_ptr(p); // expected-remark-re{{parameter 'p' of 'takes_ptr' (declared at {{.*}}) called with nonnull argument}}
}

// ===----------------------------------------------------------------------===//
// Return evidence: explicit _Nullable return -> "returns nullable"
// ===----------------------------------------------------------------------===//

int * _Nullable get_nullable();

int *return_explicit_nullable() {
    return get_nullable(); // expected-remark-re{{function 'return_explicit_nullable' of global scope (declared at {{.*}}) returns nullable}}
}

// ===----------------------------------------------------------------------===//
// Return evidence: nullptr return -> "returns nullable"
// ===----------------------------------------------------------------------===//

int *return_nullptr() {
    return nullptr; // expected-remark-re{{function 'return_nullptr' of global scope (declared at {{.*}}) returns nullable}}
}

// ===----------------------------------------------------------------------===//
// Return evidence: unannotated pointer return -> NO evidence emitted
// ===----------------------------------------------------------------------===//

int *get_unannotated();

int *return_unannotated() {
    return get_unannotated(); // no remark — return value is null-unspecified
}

// ===----------------------------------------------------------------------===//
// Return evidence: _Nonnull return -> "returns nonnull" (unchanged)
// ===----------------------------------------------------------------------===//

int *return_nonnull(int * _Nonnull p) { // expected-remark{{function 'return_nonnull' always returns a non-null pointer}}
    return p; // expected-remark-re{{function 'return_nonnull' of global scope (declared at {{.*}}) returns nonnull}}
}

// ===----------------------------------------------------------------------===//
// Member evidence: assignment from unannotated -> NO evidence
// ===----------------------------------------------------------------------===//

struct S {
    int *field;
};

void assign_unannotated(S *_Nonnull s, int *p) {
    s->field = p; // no remark — p is null-unspecified
}

// ===----------------------------------------------------------------------===//
// Member evidence: assignment from _Nullable -> "assigned from nullable"
// ===----------------------------------------------------------------------===//

void assign_nullable(S *_Nonnull s, int * _Nullable p) {
    s->field = p; // expected-remark-re{{member 'field' of S (declared at {{.*}}) assigned from nullable source}}
}

// ===----------------------------------------------------------------------===//
// Member evidence: assignment from nullptr -> "assigned from nullable"
// ===----------------------------------------------------------------------===//

void assign_nullptr(S *_Nonnull s) {
    s->field = nullptr; // expected-remark-re{{member 'field' of S (declared at {{.*}}) assigned from nullable source}}
}

// ===----------------------------------------------------------------------===//
// Member evidence: assignment from _Nonnull -> "assigned from nonnull"
// ===----------------------------------------------------------------------===//

void assign_nonnull(S *_Nonnull s, int * _Nonnull p) {
    s->field = p; // expected-remark-re{{member 'field' of S (declared at {{.*}}) assigned from nonnull source}}
}
