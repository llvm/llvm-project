// Tests the per-function suppression of the legacy type-based
// nullable->nonnull conversion warning (warn_nullability_lost,
// -Wnullable-to-nonnull-conversion) when flow-sensitive nullability is enabled.
//
// The legacy warning is suppressed inside any function: under
// -fnullability-default=unspecified a function opts in through its signature
// or through the annotated declarations its body uses, and the conversion's
// own _Nullable source and _Nonnull destination are such uses. So:
//   (a) an annotated function -> legacy warning suppressed, the flow analysis
//       covers the conversion instead;
//   (b) an unannotated function using annotated declarations -> opted in by
//       its body, so again only the flow analysis reports;
//   (c) outside a function (a global initializer) the legacy warning fires.
//
// -Wnullable-to-nonnull-conversion is DefaultIgnore, so we enable it explicitly.
//
// RUN: %clang_cc1 -fsyntax-only -fnullability-safety -std=c++17 -Wnullable-to-nonnull-conversion %s -verify

// ===----------------------------------------------------------------------===//
// (a) Opted-in function (explicit annotation): legacy warning suppressed.
//     The conversion from nullable p to a _Nonnull target is instead handled
//     by the flow analysis, which warns about the assignment.
// ===----------------------------------------------------------------------===//

void opted_in(int * _Nullable p) {
  // The flow analysis reports this nullable->nonnull assignment. Because -verify
  // fails on any unexpected diagnostic, the absence of the legacy
  // warn_nullability_lost text ("implicit conversion from nullable pointer")
  // here is itself the assertion that suppression fired for this analyzed
  // function.
  int * _Nonnull q = p; // expected-warning{{assigning nullable pointer to nonnull variable}} expected-note{{add a null check before assigning, or change the variable type to '_Nullable'}}
  (void)q;
}

// ===----------------------------------------------------------------------===//
// (b) Unannotated function: nullable_src() and the local are annotated, so the
//     body opts the function in and the flow analysis reports the store.
// ===----------------------------------------------------------------------===//

int * _Nullable nullable_src();
int * _Nonnull nonnull_src();

void opted_in_by_body() {
  int * _Nonnull nonnull = nonnull_src(); // nonnull init: no warning
  nonnull = nullable_src(); // expected-warning{{assigning nullable pointer to nonnull variable}} expected-note{{add a null check before assigning}}
  (void)nonnull;
}

// (c) No enclosing function: nothing else reports the conversion.
int * _Nonnull global_nonnull = nullable_src(); // expected-warning{{implicit conversion from nullable pointer 'int * _Nullable' to non-nullable pointer type 'int * _Nonnull'}}
