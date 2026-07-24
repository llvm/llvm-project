// Negative driver tests for nullsafe flags.
// Verifies flag interaction patterns and valid/invalid combinations.

// === -fflow-sensitive-nullability without -fnullability-default ===
// Should be accepted — defaults to unspecified.
// RUN: %clang -### -fflow-sensitive-nullability %s 2>&1 | FileCheck -check-prefix=FLOW-ONLY %s
// FLOW-ONLY: "-fflow-sensitive-nullability"

// === All three valid values for -fnullability-default ===
// RUN: %clang -### -fnullability-default=nullable %s 2>&1 | FileCheck -check-prefix=NULLABLE %s
// RUN: %clang -### -fnullability-default=nonnull %s 2>&1 | FileCheck -check-prefix=NONNULL %s
// RUN: %clang -### -fnullability-default=unspecified %s 2>&1 | FileCheck -check-prefix=UNSPEC %s
// NULLABLE: "-fnullability-default=nullable"
// NONNULL: "-fnullability-default=nonnull"
// UNSPEC: "-fnullability-default=unspecified"

// === Invalid -fnullability-default value is passed through to cc1 ===
// RUN: %clang -### -fnullability-default=invalid %s 2>&1 | FileCheck -check-prefix=INVALID %s
// INVALID: "-fnullability-default=invalid"

// === cc1 rejects invalid -fnullability-default value ===
// (tested in Sema/flow-nullability-warning-groups.cpp — cc1 tests can't live in Driver/)

// === -fno-flow-sensitive-nullability disables the flag ===
// RUN: %clang -### -fflow-sensitive-nullability -fno-flow-sensitive-nullability %s 2>&1 | FileCheck -check-prefix=NO-FLOW %s
// NO-FLOW-NOT: "-fflow-sensitive-nullability"

// === All flags together ===
// RUN: %clang -### -fflow-sensitive-nullability -fnullability-default=nullable %s 2>&1 | FileCheck -check-prefix=ALL %s
// ALL: "-fflow-sensitive-nullability"
// ALL: "-fnullability-default=nullable"
