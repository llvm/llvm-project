// RUN: %clang -target x86_64 -S -c -O1 -fvectorize -emit-llvm -o - %s | FileCheck %s -check-prefix=CHECK-ENABLE-VECT
// RUN: %clang -target x86_64 -S -c -O1 -fno-vectorize -emit-llvm -o - %s | FileCheck %s -check-prefix=CHECK-DISABLE-VECT
// RUN: %clang -target x86_64 -fexperimental-new-pass-manager -S -c -O1 -fvectorize -emit-llvm -o - %s | FileCheck %s -check-prefix=CHECK-ENABLE-VECT
// RUN: %clang -target x86_64 -fexperimental-new-pass-manager -S -c -O1 -fno-vectorize -emit-llvm -o - %s | FileCheck %s -check-prefix=CHECK-DISABLE-VECT

// CHECK-ENABLE-VECT-LABEL: @for_test()
// CHECK-ENABLE-VECT: fmul <{{[0-9]+}} x double>

// CHECK-DISABLE-VECT-LABEL: @for_test()
// CHECK-DISABLE-VECT: fmul double
// CHECK-DISABLE-VECT-NOT: fmul <{{[0-9]+}} x double>

#include <stdio.h>

void for_test() {
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
