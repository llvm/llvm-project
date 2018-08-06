// REQUIRES: x86-registered-target

// RUN: %clang_cc1 -triple x86_64-unknown-unknown %s -S -o - | FileCheck %s --check-prefix=CHECK-ABSENT
// CHECK-ABSENT-NOT: section .stack_sizes

// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fstack-size-section %s -S -o - | FileCheck %s --check-prefix=CHECK-PRESENT
// CHECK-PRESENT: section .stack_sizes

int foo() { return 42; }
