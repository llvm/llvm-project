// Tests that evidence emission distinguishes explicit _Nullable from
// null-unspecified (unannotated) sources. Unannotated pointers defaulted to
// nullable by -fnullability-default=nullable should NOT produce "nullable"
// evidence — only explicitly _Nullable or nullptr should.
//
// RUN: %clang_cc1 -fsyntax-only -fnullability-safety -fnullability-default=nullable -Wno-nullable-to-nonnull-conversion -std=c++17 %s -verify
// RUN: rm -f %t.json
// RUN: %clang_cc1 -fsyntax-only -fnullability-default=nullable -std=c++17 %s \
// RUN:   --ssaf-extract-summaries=NullabilitySafety \
// RUN:   --ssaf-compilation-unit-id=tu --ssaf-tu-summary-file=%t.json
// RUN: %python %S/../Analysis/Scalable/NullabilitySafety/Inputs/decode-summary.py %t.json | FileCheck %s --check-prefix=EVIDENCE

// ===----------------------------------------------------------------------===//
// Parameter evidence: explicit _Nullable argument -> "called with nullable"
// ===----------------------------------------------------------------------===//

void takes_ptr(int *p);

void pass_explicit_nullable(int * _Nullable np) {
    takes_ptr(np);
}

// ===----------------------------------------------------------------------===//
// Parameter evidence: nullptr -> "called with nullable"
// ===----------------------------------------------------------------------===//

void pass_nullptr() {
    takes_ptr(nullptr);
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
    takes_ptr(p);
}

// ===----------------------------------------------------------------------===//
// Return evidence: explicit _Nullable return -> "returns nullable"
// ===----------------------------------------------------------------------===//

int * _Nullable get_nullable();

int *return_explicit_nullable() {
    return get_nullable();
}

// ===----------------------------------------------------------------------===//
// Return evidence: nullptr return -> "returns nullable"
// ===----------------------------------------------------------------------===//

int *return_nullptr() {
    return nullptr;
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

int *return_nonnull(int * _Nonnull p) {
    return p;
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
    s->field = p;
}

// ===----------------------------------------------------------------------===//
// Member evidence: assignment from nullptr -> "assigned from nullable"
// ===----------------------------------------------------------------------===//

void assign_nullptr(S *_Nonnull s) {
    s->field = nullptr;
}

// ===----------------------------------------------------------------------===//
// Member evidence: assignment from _Nonnull -> "assigned from nonnull"
// ===----------------------------------------------------------------------===//

void assign_nonnull(S *_Nonnull s, int * _Nonnull p) {
    s->field = p;
}

// ===----------------------------------------------------------------------===//
// Parameter evidence: nonnull-parameter narrowing runs before evidence
// ===----------------------------------------------------------------------===//
// Only the second parameter is _Nonnull, but surviving the call proves p
// non-null, so the evidence for the first parameter must be nonnull too.
// If evidence were emitted before the narrowing pass, 'a' would get no
// remark at all (p is null-unspecified).

void two_params(int *a, int *_Nonnull b);

void call_two_params(int *p) {
    two_params(p, p); // expected-warning{{passing nullable pointer to nonnull parameter}} expected-note{{add a null check before the call}}
}

// ===----------------------------------------------------------------------===//
// Flow taint on a _Nonnull local is provable nullable evidence
// ===----------------------------------------------------------------------===//
// The declared _Nonnull does not win over an assignment of null: the argument
// and return evidence say nullable, and no all-returns-nonnull summary is
// inferred.

void pass_tainted_nonnull() {
    int *_Nonnull a = nullptr; // expected-warning{{initializing a variable of nonnull type}}
    takes_ptr(a);
}

int *return_tainted_nonnull() {
    int *_Nonnull a = nullptr; // expected-warning{{initializing a variable of nonnull type}}
    return a;
}

// EVIDENCE-NOT:  {{.}}
// EVIDENCE:      c:@F@assign_nonnull#*$@S@S#*I# NonnullEvidence c:@S@S@FI@field
// EVIDENCE-NEXT: c:@F@assign_nullable#*$@S@S#*I# NullableEvidence c:@S@S@FI@field
// EVIDENCE-NEXT: c:@F@assign_nullptr#*$@S@S# NullableEvidence c:@S@S@FI@field
// EVIDENCE-NEXT: c:@F@assign_unannotated#*$@S@S#*I# ConditionalEvidence c:@S@S@FI@field <- c:@F@assign_unannotated#*$@S@S#*I# param 2
// EVIDENCE-NEXT: c:@F@call_two_params#*I# NonnullEvidence c:@F@two_params#*I#S0_# param 1
// EVIDENCE-NEXT: c:@F@call_two_params#*I# NonnullEvidence c:@F@two_params#*I#S0_# param 2
// EVIDENCE-NEXT: c:@F@pass_explicit_nullable#*I# NullableEvidence c:@F@takes_ptr#*I# param 1
// EVIDENCE-NEXT: c:@F@pass_nonnull#*I# NonnullEvidence c:@F@takes_ptr#*I# param 1
// EVIDENCE-NEXT: c:@F@pass_nullptr# NullableEvidence c:@F@takes_ptr#*I# param 1
// EVIDENCE-NEXT: c:@F@pass_tainted_nonnull# NullableEvidence c:@F@takes_ptr#*I# param 1
// EVIDENCE-NEXT: c:@F@pass_unannotated#*I# ConditionalEvidence c:@F@takes_ptr#*I# param 1 <- c:@F@pass_unannotated#*I# param 1
// EVIDENCE-NEXT: c:@F@return_explicit_nullable# NullableEvidence c:@F@return_explicit_nullable# return
// EVIDENCE-NEXT: c:@F@return_nonnull#*I# AllReturnsNonnull c:@F@return_nonnull#*I# return
// EVIDENCE-NEXT: c:@F@return_nonnull#*I# NonnullEvidence c:@F@return_nonnull#*I# return
// EVIDENCE-NEXT: c:@F@return_nullptr# NullableEvidence c:@F@return_nullptr# return
// EVIDENCE-NEXT: c:@F@return_tainted_nonnull# NullableEvidence c:@F@return_tainted_nonnull# return
// EVIDENCE-NEXT: c:@F@return_unannotated# ConditionalEvidence c:@F@return_unannotated# return <- c:@F@get_unannotated# return
// EVIDENCE-NOT:  {{.}}
