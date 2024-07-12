// RUN: %clang -fsyntax-only -S %s 2>&1 | FileCheck %s --check-prefix=CHECK-ASM
// RUN: %clang -fsyntax-only -c %s 2>&1 | FileCheck %s --check-prefix=CHECK-OBJ
// RUN: %clang -fsyntax-only -S -c %s 2>&1 | FileCheck %s --check-prefix=CHECK-BOTH

// CHECK-ASM: warning: argument unused during compilation: '-S' [-Wunused-command-line-argument]
// CHECK-OBJ: warning: argument unused during compilation: '-c' [-Wunused-command-line-argument]

// CHECK-BOTH: warning: argument unused during compilation: '-S' [-Wunused-command-line-argument]
// CHECK-NEXT: warning: argument unused during compilation: '-c' [-Wunused-command-line-argument]
