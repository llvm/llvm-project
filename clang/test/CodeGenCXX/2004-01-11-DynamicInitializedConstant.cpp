// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s

// Catch the beginning and the end of the IR. This prevents the CHECK-NOT from
// matching a spurious "constant" string in file paths printed in the output.
//
// CHECK-LABEL: target triple
// CHECK-NOT: constant
// CHECK-LABEL: attributes

extern int X;
const int Y = X;
const int* foo() { return &Y; }
