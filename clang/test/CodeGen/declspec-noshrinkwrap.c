// RUN: %clang_cc1 -fdeclspec -emit-llvm %s -o - | FileCheck %s

// Ensure that declspec results in a llvm function attribute
__declspec(noshrinkwrap)
int square(int a) {
  return a*a;
}

// CHECK: Function Attrs: {{.*}} noshrinkwrap
// CHECK-LABEL: @square
// CHECK: attributes #0 {{.*}} noshrinkwrap
