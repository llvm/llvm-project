// RUN: %clang --target=x86_64 -fsyntax-only -E %s 2>&1 | FileCheck %s --check-prefix=CHECK-PP
// RUN: %clang --target=x86_64 -fsyntax-only -S %s 2>&1 | FileCheck %s --check-prefix=CHECK-ASM
// RUN: %clang --target=x86_64 -fsyntax-only -c %s 2>&1 | FileCheck %s --check-prefix=CHECK-OBJ

// CHECK-PP:  warning: argument unused during compilation: '-fsyntax-only' [-Wunused-command-line-argument]
// CHECK-ASM: warning: argument unused during compilation: '-S' [-Wunused-command-line-argument]
// CHECK-OBJ: warning: argument unused during compilation: '-c' [-Wunused-command-line-argument]
