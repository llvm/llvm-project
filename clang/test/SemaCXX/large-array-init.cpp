// RUN: %clang_cc1 -S -o %t.ll -mllvm -debug-only=exprconstant                                         %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -S -o %t.ll -mllvm -debug-only=exprconstant -fexperimental-new-constant-interpreter %s 2>&1 | FileCheck %s
// REQUIRES: asserts

struct S { int i; };

static struct S arr[100000000] = {{ 0 }};
// CHECK: The number of elements to initialize: 1.

struct S *foo() { return arr; }
