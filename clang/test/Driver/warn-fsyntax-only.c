// RUN: %clang -fsyntax-only -E %s 2>&1 | FileCheck %s --check-prefix=CHECK-PP
// RUN: %clang -fsyntax-only -S %s 2>&1 | FileCheck %s --check-prefix=CHECK-ASM
// RUN: %clang -fsyntax-only -c %s 2>&1 | FileCheck %s --check-prefix=CHECK-OBJ

// CHECK-PP:  warning: argument unused during compilation: '-fsyntax-only' [-Wunused-command-line-argument]
// CHECK-ASM: warning: argument unused during compilation: '-S' [-Wunused-command-line-argument]
// CHECK-OBJ: warning: argument unused during compilation: '-c' [-Wunused-command-line-argument]
