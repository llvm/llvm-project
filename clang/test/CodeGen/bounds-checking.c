// RUN: %clang_cc1 -fbounds-checking=4 -emit-llvm -triple x86_64-apple-darwin10 < %s | FileCheck %s

// CHECK: @f
double f(int b, int i) {
  double a[b];
  // CHECK: trap
  return a[i];
}

// CHECK: @f2
void f2() {
  // everything is constant; no trap possible
  // CHECK-NOT: trap
  int a[2];
  a[1] = 42;
  
  short *b = malloc(64);
  b[5] = *a + a[1] + 2;
}

// CHECK: @f3
void f3() {
  int a[1];
  // CHECK: trap
  a[2] = 1;
}
