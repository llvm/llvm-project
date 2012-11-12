// REQUIRES: shell
// RUN: ln -sf %clang %T/clang-cpp

// PR13529: Don't crash.
// RUN: %T/clang-cpp -lfoo -M %s 2>&1 | FileCheck --check-prefix=CHECK-PR13529 %s
// CHECK-PR13529: warning: -lfoo: 'linker' input unused in cpp mode
