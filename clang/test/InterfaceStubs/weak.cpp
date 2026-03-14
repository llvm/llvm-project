// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -triple x86_64-linux-gnu -o - -emit-interface-stubs \
// RUN: -interface-stub-version=ifs-v1 %s | \
// RUN: FileCheck %s

// RUN: %clang -target x86_64-unknown-linux-gnu -o - -c %s | llvm-nm - 2>&1 | \
// RUN: FileCheck -check-prefix=CHECK-SYMBOLS %s

// CHECK: Symbols:
// CHECK-DAG:  - { Name: "_Z8weakFuncv", Type: Func, Weak: true }
// CHECK-DAG:  - { Name: "_Z10strongFuncv", Type: Func }

// CHECK-SYMBOLS-DAG: _Z10strongFuncv
// CHECK-SYMBOLS-DAG: _Z8weakFuncv
__attribute__((weak)) void weakFunc() {}
int strongFunc() { return 42; }
