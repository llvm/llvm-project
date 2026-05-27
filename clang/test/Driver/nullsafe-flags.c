// RUN: %clang -### -fflow-sensitive-nullability %s 2>&1 | FileCheck -check-prefix=FLOW %s
// RUN: %clang -### -fnullability-default=nullable %s 2>&1 | FileCheck -check-prefix=DEFAULT %s
// RUN: %clang -### -fflow-sensitive-nullability -fnullability-default=nullable %s 2>&1 | FileCheck -check-prefix=BOTH %s

// FLOW: "-fflow-sensitive-nullability"
// DEFAULT: "-fnullability-default=nullable"
// BOTH: "-fflow-sensitive-nullability"
// BOTH: "-fnullability-default=nullable"
