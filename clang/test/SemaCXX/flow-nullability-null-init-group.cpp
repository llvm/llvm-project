// warn_null_init_nonnull (null-initialization of a _Nonnull variable) is an
// initialization diagnostic, so it belongs to -Wflow-nullable-assignment, NOT
// -Wflow-nullable-dereference.
//
// It is emitted by SemaDecl purely on -fflow-sensitive-nullability (it does not
// require the function to opt in to the flow analysis). We deliberately keep
// the default nullability 'unspecified' and give the function no signature
// annotations, so the flow analysis does NOT run for it — that isolates
// warn_null_init_nonnull from the flow analysis's own
// warn_flow_nullable_assignment (which is also in the assignment group and
// would otherwise mask which group is being exercised).
//
// Baseline: the warning fires by default when the feature is on.
// RUN: %clang_cc1 -fsyntax-only -fflow-sensitive-nullability -verify=warn %s
//
// -Wno-flow-nullable-assignment silences it (its group after the move):
// RUN: %clang_cc1 -fsyntax-only -fflow-sensitive-nullability -Wno-flow-nullable-assignment -verify=silenced %s
//
// Parent group -Wno-flow-nullability also silences it:
// RUN: %clang_cc1 -fsyntax-only -fflow-sensitive-nullability -Wno-flow-nullability -verify=silenced %s
//
// -Wno-flow-nullable-dereference does NOT silence it (wrong group):
// RUN: %clang_cc1 -fsyntax-only -fflow-sensitive-nullability -Wno-flow-nullable-dereference -verify=warn %s
//
// -Werror=flow-nullable-assignment promotes it to an error:
// RUN: %clang_cc1 -fsyntax-only -fflow-sensitive-nullability -Werror=flow-nullable-assignment -verify=werror %s

// silenced-no-diagnostics

// No annotations on the signature -> not opted in -> flow analysis skipped, so
// only the type-based warn_null_init_nonnull fires here.
void nullInitNonnull() {
  int *_Nonnull p = nullptr; // warn-warning {{null assigned to a variable of nonnull type}} \
                                werror-error {{null assigned to a variable of nonnull type}}
  (void)p;
}
