// RUN: %clang_cc1 -triple aarch64 -O0 -S -o - %s | FileCheck %s --check-prefix   ALL_BUILDS
// RUN: %clang_cc1 -triple aarch64 -O1 -S -o - %s | FileCheck %s --check-prefixes ALL_BUILDS,NON_O0
// RUN: %clang_cc1 -triple aarch64 -O2 -S -o - %s | FileCheck %s --check-prefixes ALL_BUILDS,NON_O0
// RUN: %clang_cc1 -triple aarch64 -O3 -S -o - %s | FileCheck %s --check-prefixes ALL_BUILDS,NON_O0

// REQUIRES: aarch64-registered-target

// ALL_BUILDS-LABEL: fmadd_double:
// ALL_BUILDS: fmadd d0, d{{[0-7]}}, d{{[0-7]}}, d{{[0-7]}}
// NON_O0-NEXT: ret
double fmadd_double(double a, double b, double c) {
  return a*b+c;
}

// ALL_BUILDS: fmadd_single:
// ALL_BUILDS: fmadd s0, s{{[0-7]}}, s{{[0-7]}}, s{{[0-7]}}
// NON_O0-NEXT: ret
float  fmadd_single(float  a, float  b, float  c) {
  return a*b+c;
}

