// Tests that flow-sensitive nullability opt-in walks the whole redeclaration
// chain. A function whose nullability annotations live on a separate prototype
// (e.g. a header declaration) must still be analyzed even when the definition
// we hand to the analysis is itself unannotated.
//
// Without -fnullability-default the only opt-in signal is explicit annotation,
// so this exercises functionHasNullabilityAnnotations() across redecls().
// Pre-fix, opt-in inspected only the definition and silently skipped these.
//
// RUN: %clang_cc1 -fsyntax-only -fflow-sensitive-nullability -std=c++17 -Wno-nullable-to-nonnull-conversion %s -verify

// ===----------------------------------------------------------------------===//
// Annotated parameter on the prototype, unannotated definition: MUST analyze.
// The nullable param is dereferenced without a null check, so a flow warning
// fires — proving the body was analyzed.
// ===----------------------------------------------------------------------===//

void param_annotated_proto(int * _Nullable p);

void param_annotated_proto(int *p) { // definition unannotated
  *p = 0; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

// ===----------------------------------------------------------------------===//
// Annotated RETURN type only on the prototype, fully unannotated definition
// (no annotated params). This is the case parameter-type merging does NOT
// cover, so it genuinely depends on opt-in walking redecls(): the body must be
// analyzed. We prove analysis ran by dereferencing a _Nullable value without a
// check inside the body.
// ===----------------------------------------------------------------------===//

int * _Nullable nullable_helper();

int * _Nonnull return_annotated_proto();

int * _Nonnull other_nonnull();

int *return_annotated_proto() { // definition has no annotations of its own
  int *q = nullable_helper();
  *q = 0; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
  return other_nonnull(); // nonnull: no return warning
}

// ===----------------------------------------------------------------------===//
// Control: NO annotation anywhere in the chain. Under unspecified default the
// function never opts in, so flow analysis is skipped — no flow warning fires.
// ===----------------------------------------------------------------------===//

void not_opted_in(int *p);

void not_opted_in(int *p) {
  *p = 0; // no warning: function never opted in
}
