// Negative driver tests for nullability safety flags.
// Verifies flag interaction patterns and valid/invalid combinations.

// === -fnullability-safety without -fnullability-default ===
// Should be accepted — defaults to unspecified.
// RUN: %clang -### -fnullability-safety %s 2>&1 | FileCheck -check-prefix=FLOW-ONLY %s
// FLOW-ONLY: "-fnullability-safety"

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
// (tested in Sema/nullability-safety-warning-groups.cpp — cc1 tests can't live in Driver/)

// === -fno-nullability-safety disables the flag ===
// RUN: %clang -### -fnullability-safety -fno-nullability-safety %s 2>&1 | FileCheck -check-prefix=NO-FLOW %s
// NO-FLOW-NOT: "-fnullability-safety"

// === All flags together ===
// RUN: %clang -### -fnullability-safety -fnullability-default=nullable %s 2>&1 | FileCheck -check-prefix=ALL %s
// ALL: "-fnullability-safety"
// ALL: "-fnullability-default=nullable"
