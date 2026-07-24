// Tests the per-function suppression of the legacy type-based
// nullable->nonnull conversion warning (warn_nullability_lost,
// -Wnullable-to-nonnull-conversion) when flow-sensitive nullability is enabled.
//
// The legacy warning must be suppressed ONLY for functions the flow checker
// will actually analyze. Under -fnullability-default=unspecified, opt-in is by
// explicit annotation, so:
//   (a) an annotated (opted-in) function -> legacy warning suppressed, the
//       flow analysis covers the conversion instead;
//   (b) an unannotated (non-opted-in) function -> NOT analyzed, so the legacy
//       type-based warning must still fire (the coverage hole closed).
//
// -Wnullable-to-nonnull-conversion is DefaultIgnore, so we enable it explicitly.
//
// RUN: %clang_cc1 -fsyntax-only -fflow-sensitive-nullability -std=c++17 -Wnullable-to-nonnull-conversion %s -verify

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
// (b) Non-opted-in function (no annotation anywhere): the flow checker skips
//     it, so the legacy type-based warning must still fire. nullable_src() is
//     declared _Nullable, but the enclosing function has no annotations and
//     -fnullability-default is unspecified, so it does not opt in.
// ===----------------------------------------------------------------------===//

int * _Nullable nullable_src();
int * _Nonnull nonnull_src();

void not_opted_in() {
  // Enclosing function 'not_opted_in' has no nullability annotations; under
  // unspecified default it is never analyzed by the flow checker, so the
  // legacy conversion warning is the only coverage and must fire.
  int * _Nonnull nonnull = nonnull_src(); // nonnull init: no warning
  nonnull = nullable_src(); // expected-warning{{implicit conversion from nullable pointer 'int * _Nullable' to non-nullable pointer type 'int * _Nonnull'}}
  (void)nonnull;
}
