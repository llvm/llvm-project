// RUN: %clang -### -fnullability-safety %s 2>&1 | FileCheck -check-prefix=FLOW %s
// RUN: %clang -### -fnullability-default=nullable %s 2>&1 | FileCheck -check-prefix=DEFAULT %s
// RUN: %clang -### -fnullability-safety -fnullability-default=nullable %s 2>&1 | FileCheck -check-prefix=BOTH %s
// RUN: %clang -### -fnullability-safety -fno-nullability-libc-nullable-returns %s 2>&1 | FileCheck -check-prefix=NOLIBC %s
// The C library nullable-return list is on by default, so cc1 should not get any flag for it.
// RUN: %clang -### -fnullability-safety %s 2>&1 | FileCheck -check-prefix=LIBC-DEFAULT %s

// FLOW: "-fnullability-safety"
// DEFAULT: "-fnullability-default=nullable"
// BOTH: "-fnullability-safety"
// BOTH: "-fnullability-default=nullable"
// NOLIBC: "-fno-nullability-libc-nullable-returns"
// LIBC-DEFAULT-NOT: "-fno-nullability-libc-nullable-returns"
