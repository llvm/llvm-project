// RUN: %clang_cc1 -emit-llvm %s -o - | FileCheck %s
// rdar://7589850

// CHECK-NOT: __ustring
void *P = @"good\0bye";
