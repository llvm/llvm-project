// This file contain tests for attribute arguments on K&R functions.

// RUN: %clang_cc1 -ast-print -x c -std=c89 -fms-extensions %s -o - | FileCheck %s

// CHECK: int knr(i)
// CHECK-NEXT: int i __attribute__((unused));
// CHECK-NEXT: {
// CHECK-NEXT: return 0;
// CHECK-NEXT: }
int knr(i) int i __attribute__((unused)); { return 0; }

// CHECK: __attribute__((unused)) int knr2(i)
// CHECK-NEXT: int i;
// CHECK-NEXT: {
// CHECK-NEXT: return 0;
// CHECK-NEXT: }
__attribute__((unused)) int knr2(i) int i; { return 0; }
