// RUN: %clang_cc1 -verify -fopenmp -x c -emit-llvm %s -triple x86_64-unknown-linux -o - -femit-all-decls -disable-llvm-passes -fprofile-sample-use=prof | FileCheck %s
// expected-no-diagnostics

// CHECK: define {{.*}}@sub.omp_outlined{{.*}} #[[ATTR:[0-9]+]]
// CHECK: attributes #[[ATTR]] = {{.*}}"sample-profile-suffix-elision-policy"="selected"

void sub(double *restrict a, double *restrict b, int n) {
  int i;

#pragma omp parallel for
  for (i = 0; i < n; i++) {
    a[i] = a[i] + b[i];
  }
}
