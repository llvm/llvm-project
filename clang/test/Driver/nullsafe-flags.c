// RUN: %clang -### -fflow-sensitive-nullability %s 2>&1 | FileCheck -check-prefix=FLOW %s
// RUN: %clang -### -fnullability-default=nullable %s 2>&1 | FileCheck -check-prefix=DEFAULT %s
// RUN: %clang -### -fflow-sensitive-nullability -fnullability-default=nullable %s 2>&1 | FileCheck -check-prefix=BOTH %s
// RUN: %clang -### -fflow-sensitive-nullability -fno-nullability-stdlib-annotations %s 2>&1 | FileCheck -check-prefix=NOSTDLIB %s
// The stdlib annotation list is on by default, so cc1 should not get any flag for it.
// RUN: %clang -### -fflow-sensitive-nullability %s 2>&1 | FileCheck -check-prefix=STDLIB-DEFAULT %s

// FLOW: "-fflow-sensitive-nullability"
// DEFAULT: "-fnullability-default=nullable"
// BOTH: "-fflow-sensitive-nullability"
// BOTH: "-fnullability-default=nullable"
// NOSTDLIB: "-fno-nullability-stdlib-annotations"
// STDLIB-DEFAULT-NOT: "-fno-nullability-stdlib-annotations"
