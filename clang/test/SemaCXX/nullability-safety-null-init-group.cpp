// warn_nullability_safety_null_init (null-initialization of a _Nonnull variable) is an
// initialization diagnostic, so it belongs to -Wnullability-safety-assignment, NOT
// -Wnullability-safety-dereference.
//
// It is emitted by SemaDecl purely on -fnullability-safety (it does not
// require the function to opt in to the flow analysis). We deliberately keep
// the default nullability 'unspecified' and give the function no signature
// annotations, so the flow analysis does NOT run for it — that isolates
// warn_nullability_safety_null_init from the flow analysis's own
// warn_nullability_safety_assignment (which is also in the assignment group and
// would otherwise mask which group is being exercised).
//
// Baseline: the warning fires by default when the feature is on.
// RUN: %clang_cc1 -fsyntax-only -fnullability-safety -verify=warn %s
//
// -Wno-nullability-safety-assignment silences it (its group after the move):
// RUN: %clang_cc1 -fsyntax-only -fnullability-safety -Wno-nullability-safety-assignment -verify=silenced %s
//
// Parent group -Wno-nullability-safety also silences it:
// RUN: %clang_cc1 -fsyntax-only -fnullability-safety -Wno-nullability-safety -verify=silenced %s
//
// -Wno-nullability-safety-dereference does NOT silence it (wrong group):
// RUN: %clang_cc1 -fsyntax-only -fnullability-safety -Wno-nullability-safety-dereference -verify=warn %s
//
// -Werror=nullability-safety-assignment promotes it to an error:
// RUN: %clang_cc1 -fsyntax-only -fnullability-safety -Werror=nullability-safety-assignment -verify=werror %s

// silenced-no-diagnostics

// No annotations on the signature -> not opted in -> flow analysis skipped, so
// only the type-based warn_nullability_safety_null_init fires here.
void nullInitNonnull() {
  int *_Nonnull p = nullptr; // warn-warning {{initializing a variable of nonnull type}} \
                                werror-error {{initializing a variable of nonnull type}}
  (void)p;
}
