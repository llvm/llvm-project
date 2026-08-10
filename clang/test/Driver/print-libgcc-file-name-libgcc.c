// Test that -print-libgcc-file-name correctly respects -rtlib=libgcc.

// REQUIRES: libgcc

// RUN: %clang -rtlib=libgcc -print-libgcc-file-name 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-LIBGCC %s
// CHECK-LIBGCC: libgcc.a
