// RUN: %clang_cc1 -triple x86_64 -target-cpu x86-64 -S -O1 -vectorize-loops -emit-llvm -o - %s | FileCheck %s -check-prefix=CHECK-ENABLE-VECT
// RUN: %clang_cc1 -triple x86_64 -target-cpu x86-64 -S -O1 -emit-llvm -o - %s | FileCheck %s -check-prefix=CHECK-DISABLE-VECT
// REQUIRES: x86-registered-target

// CHECK-ENABLE-VECT-LABEL: @for_test()
// CHECK-ENABLE-VECT: fmul <{{[0-9]+}} x double>

// CHECK-DISABLE-VECT-LABEL: @for_test()
// CHECK-DISABLE-VECT: fmul double
// CHECK-DISABLE-VECT-NOT: fmul <{{[0-9]+}} x double>

int printf(const char * restrict format, ...);

void for_test(void) {
  double A[1000], B[1000];
  int L = 500;
  for (int i = 0; i < L; i++) {
    A[i] = i;
  }
  for (int i = 0; i < L; i++) {
    B[i] = A[i]*5;
  }
  printf("%lf %lf\n", A[0], B[0]);
}
