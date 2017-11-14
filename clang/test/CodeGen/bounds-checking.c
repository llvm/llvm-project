// RUN: %clang_cc1 -fsanitize=local-bounds -emit-llvm -triple x86_64-apple-darwin10 %s -o - | FileCheck %s
// RUN: %clang_cc1 -fsanitize=local-bounds -fexperimental-new-pass-manager -emit-llvm -triple x86_64-apple-darwin10 %s -o - | FileCheck %s
// RUN: %clang_cc1 -fsanitize=array-bounds -O -fsanitize-trap=array-bounds -emit-llvm -triple x86_64-apple-darwin10 -DNO_DYNAMIC %s -o - | FileCheck %s
// RUN: %clang_cc1 -fsanitize=array-bounds -O -fsanitize-trap=array-bounds -fexperimental-new-pass-manager -emit-llvm -triple x86_64-apple-darwin10 -DNO_DYNAMIC %s -o - | FileCheck %s

// CHECK-LABEL: @f
double f(int b, int i) {
  double a[b];
  // CHECK: call {{.*}} @llvm.trap
  return a[i];
}

// CHECK-LABEL: @f2
void f2() {
  // everything is constant; no trap possible
  // CHECK-NOT: call {{.*}} @llvm.trap
  int a[2];
  a[1] = 42;

#ifndef NO_DYNAMIC
  short *b = malloc(64);
  b[5] = *a + a[1] + 2;
#endif
}

// CHECK-LABEL: @f3
void f3() {
  int a[1];
  // CHECK: call {{.*}} @llvm.trap
  a[2] = 1;
}
