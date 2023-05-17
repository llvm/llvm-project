// RUN: %clang_cc1 -fsanitize=local-bounds -emit-llvm -triple x86_64-apple-darwin10 %s -o - | FileCheck %s
// RUN: %clang_cc1 -fsanitize=array-bounds -O -fsanitize-trap=array-bounds -emit-llvm -triple x86_64-apple-darwin10 -DNO_DYNAMIC %s -o - | FileCheck %s
//
// REQUIRES: x86-registered-target

// CHECK-LABEL: @f1
double f1(int b, int i) {
  double a[b];
  // CHECK: call {{.*}} @llvm.{{(ubsan)?trap}}
  return a[i];
}

// CHECK-LABEL: @f2
void f2(void) {
  // everything is constant; no trap possible
  // CHECK-NOT: call {{.*}} @llvm.{{(ubsan)?trap}}
  int a[2];
  a[1] = 42;

#ifndef NO_DYNAMIC
  extern void *malloc(__typeof__(sizeof(0)));
  short *b = malloc(64);
  b[5] = *a + a[1] + 2;
#endif
}

// CHECK-LABEL: @f3
void f3(void) {
  int a[1];
  // CHECK: call {{.*}} @llvm.{{(ubsan)?trap}}
  a[2] = 1;
}

// CHECK-LABEL: @f4
__attribute__((no_sanitize("bounds")))
int f4(int i) {
  int b[64];
  // CHECK-NOT: call void @llvm.trap()
  // CHECK-NOT: trap:
  // CHECK-NOT: cont:
  return b[i];
}

// Union flexible-array memebers are a C99 extension. All array members with a
// constant size should be considered FAMs.

union U { int a[0]; int b[1]; int c[2]; };

// CHECK-LABEL: @f5
int f5(union U *u, int i) {
  // a is treated as a flexible array member.
  // CHECK-NOT: @llvm.ubsantrap
  return u->a[i];
}

// CHECK-LABEL: @f6
int f6(union U *u, int i) {
  // b is treated as a flexible array member.
  // CHECK-NOT: call {{.*}} @llvm.{{(ubsan)?trap}}
  return u->b[i];
}

// CHECK-LABEL: @f7
int f7(union U *u, int i) {
  // c is treated as a flexible array member.
  // CHECK-NOT: @llvm.ubsantrap
  return u->c[i];
}
