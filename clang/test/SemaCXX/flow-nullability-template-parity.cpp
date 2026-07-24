// Template-instantiation parity for flow-sensitive nullability.
//
// Goal: the analysis should behave IDENTICALLY for hand-written code and for the
// same code produced by instantiating a function template. Divergence would mean
// nullability sugar applied to an instantiated declarator differs from the
// equivalent non-template declarator.
//
// Background: default-nullability injection in SemaType.cpp is guarded by
// `S.CodeSynthesisContexts.empty()`, so unannotated pointers in an INSTANTIATED
// declarator are NOT tagged with `-fnullability-default`, while the equivalent
// non-template declarator IS. The unannotated cases below assert the current
// (divergent) behavior with a FIXME so this file passes today; the annotated
// cases assert the desired parity, which already holds.
//
// RUN: %clang_cc1 -fsyntax-only -fflow-sensitive-nullability -fnullability-default=nullable -std=c++17 -Wno-unused-value %s -verify

// NB: deliberately NOT inside a `#pragma clang assume_nonnull` region — that
// would force unannotated pointers to _Nonnull and mask the
// -fnullability-default=nullable behavior that Case 2 exercises.

// ==========================================================================
// Case 1: EXPLICITLY annotated _Nullable parameter.
// The annotation is on the written type and survives substitution, so the
// template instantiation and the hand-written function must both warn.
// This parity already holds.
// ==========================================================================

// Hand-written.
int deref_nontemplate(int *_Nullable p) {
  return *p; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

// Template, instantiated below with T = int.
template <class T>
int deref_template(T *_Nullable p) {
  return *p; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

void instantiate_annotated(int *_Nullable p) {
  deref_template<int>(p); // forces instantiation -> warning above must fire
}

// ==========================================================================
// Case 2: UNANNOTATED pointer under -fnullability-default=nullable.
// This is where the CodeSynthesisContexts skip causes divergence.
//
// Non-template: the unannotated `int *q` is tagged _Nullable by default
// injection, so the dereference warns.
//
// Template: during instantiation CodeSynthesisContexts is non-empty, so the
// unannotated `T *q` is NOT tagged _Nullable, and the dereference does NOT
// warn. This is the real divergence.
// ==========================================================================

// Hand-written: default injection applies -> warns.
int deref_unannotated_nontemplate(int *q) {
  return *q; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

// Template: default injection is skipped during instantiation.
template <class T>
int deref_unannotated_template(T *q) {
  // FIXME: should warn{{dereference of nullable pointer}} to match
  // deref_unannotated_nontemplate. It does NOT, because -fnullability-default
  // injection is gated on `S.CodeSynthesisContexts.empty()` and is therefore
  // skipped while instantiating this template. Asserting the current (no
  // warning) behavior so the test passes and the divergence stays documented.
  return *q; // no warning (DIVERGENCE from non-template form)
}

void instantiate_unannotated(int *q) {
  deref_unannotated_template<int>(q); // forces instantiation
}
