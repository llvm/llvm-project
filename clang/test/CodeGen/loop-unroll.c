// RUN: %clang_cc1 -triple x86_64 -target-cpu x86-64 -S -O1 -funroll-loops -emit-llvm -o - %s | FileCheck %s -check-prefix=CHECK-ENABLE-UNROLL
// RUN: %clang_cc1 -triple x86_64 -target-cpu x86-64 -S -O1 -fno-unroll-loops -emit-llvm -o - %s | FileCheck %s -check-prefix=CHECK-DISABLE-UNROLL
// REQUIRES: x86-registered-target

// CHECK-ENABLE-UNROLL-LABEL: @for_test()
// CHECK-ENABLE-UNROLL: br label %[[FORBODY:[a-z0-9_\.]+]]
// CHECK-ENABLE-UNROLL: [[FORBODY]]:
// CHECK-ENABLE-UNROLL: store
// CHECK-ENABLE-UNROLL: store
// CHECK-ENABLE-UNROLL: br i1 %[[EXITCOND:[a-z0-9_\.]+]], label %[[FORBODY5:[a-z0-9_\.]+]], label %[[FORBODY]]
// CHECK-ENABLE-UNROLL: [[FORBODY5]]:
// CHECK-ENABLE-UNROLL: fmul
// CHECK-ENABLE-UNROLL: fadd
// CHECK-ENABLE-UNROLL: store
// CHECK-ENABLE-UNROLL: fmul
// CHECK-ENABLE-UNROLL: fadd
// CHECK-ENABLE-UNROLL: store
// CHECK-ENABLE-UNROLL: fmul
// CHECK-ENABLE-UNROLL: fadd
// CHECK-ENABLE-UNROLL: store

// CHECK-DISABLE-UNROLL-LABEL: @for_test()
// CHECK-DISABLE-UNROLL: br label %[[FORBODY:[a-z0-9_\.]+]]
// CHECK-DISABLE-UNROLL: [[FORBODY]]:
// CHECK-DISABLE-UNROLL: store
// CHECK-DISABLE-UNROLL-NOT: store
// CHECK-DISABLE-UNROLL: br i1 %[[EXITCOND:[a-z0-9_\.]+]], label %[[FORBODY5:[a-z0-9_\.]+]], label %[[FORBODY]]
// CHECK-DISABLE-UNROLL: [[FORBODY5]]:
// CHECK-DISABLE-UNROLL: fmul
// CHECK-DISABLE-UNROLL: fadd
// CHECK-DISABLE-UNROLL: store
// CHECK-DISABLE-UNROLL: fmul
// CHECK-DISABLE-UNROLL: fadd
// CHECK-DISABLE-UNROLL: store
// CHECK-DISABLE-UNROLL-NOT: fmul
// CHECK-DISABLE-UNROLL-NOT: fadd
// CHECK-DISABLE-UNROLL-NOT: store
// Limit scope of checking so this does not match "fadd" within git version string
// CHECK-DISABLE-UNROLL: !0 =

int printf(const char * restrict format, ...);

void for_test(void) {
  double A[1000], B[1000];
  int L = 500;
  for (int i = 0; i < L; i++) {
    A[i] = i;
  }
  for (int i = 0; i < L; i++) {
    B[i] = A[i]*5;
    B[i]++;
    A[i] *= 7;
    A[i]++;
  }
  printf("%lf %lf\n", A[0], B[0]);
}
