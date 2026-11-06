// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O0 -fopenmp -fmath-errno \
// RUN:   -fno-builtin -disable-llvm-passes -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefixes=CHECK,ERRNO
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O0 -fopenmp -ffast-math \
// RUN:   -disable-llvm-passes -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefixes=CHECK,FAST

// Check that a call to a declare-simd libm math function inside an OpenMP SIMD
// loop is marked memory(none) and no-builtin, and that calls outside a SIMD loop
// and calls to functions with output pointers (sincos) are left alone.

#pragma omp declare simd notinbranch
double acosh(double);

#pragma omp declare simd notinbranch linear(s) linear(c)
void sincos(double, double *s, double *c);

#define N 128
double a[N], b[N];

// CHECK-LABEL: define{{.*}}@test_simd(
// CHECK: call{{.*}}double @acosh({{.*}}) [[SIMD_ATTR:#[0-9]+]]
void test_simd(void) {
#pragma omp simd
  for (int i = 0; i < N; ++i)
    a[i] = acosh(b[i]);
}

// CHECK-LABEL: define{{.*}}@test_scalar(
// ERRNO: call{{.*}}double @acosh({{.*}}) [[SCALAR_ATTR:#[0-9]+]]
double test_scalar(double x) { return acosh(x); }

// CHECK-LABEL: define{{.*}}@test_sincos_simd(
// CHECK: call{{.*}}void @sincos({{.*}}ptr{{.*}}ptr{{.*}}) [[SINCOS_ATTR:#[0-9]+]]
void test_sincos_simd(double *s, double *c) {
#pragma omp simd
  for (int i = 0; i < N; ++i)
    sincos(b[i], &s[i], &c[i]);
}

// The call site inside the SIMD loop is memory(none) and no-builtin.
// CHECK: attributes [[SIMD_ATTR]] = { {{(nobuiltin.*memory\(none\)|memory\(none\).*nobuiltin)}}
// A call to a function with output pointers is never memory(none).
// CHECK-NOT: attributes [[SINCOS_ATTR]] = {{.*}}memory\(none\)
// Under -fmath-errno the call outside a SIMD loop is not tightened either.
// ERRNO-NOT: attributes [[SCALAR_ATTR]] = {{.*}}memory\(none\)
