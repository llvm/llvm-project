// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s

// CHECK-NOT: constant
extern int X;
const int Y = X;
const int* foo() { return &Y; }
