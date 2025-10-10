// RUN: %clang -fbounds-safety -### %s 2>&1 | FileCheck --check-prefix=DEFAULT %s

// Simple use of new flag
// RUN: %clang -fbounds-safety -fbounds-safety-debug-trap-reasons=detailed -### %s 2>&1 | FileCheck --check-prefix=DETAILED %s
// RUN: %clang -fbounds-safety -fbounds-safety-debug-trap-reasons=basic -### %s 2>&1 | FileCheck --check-prefix=BASIC %s
// RUN: %clang -fbounds-safety -fbounds-safety-debug-trap-reasons=none -### %s 2>&1 | FileCheck --check-prefix=NONE %s

// DEFAULT-NOT: -fbounds-safety-debug-trap-reasons=
// DETAILED: -fbounds-safety-debug-trap-reasons=detailed
// BASIC: -fbounds-safety-debug-trap-reasons=basic
// NONE: -fbounds-safety-debug-trap-reasons=none
