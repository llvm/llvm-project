// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O0 -fopenmp -fmath-errno \
// RUN:   -fno-builtin -disable-llvm-passes -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefixes=CHECK,ERRNO
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O0 -fopenmp -ffast-math \
// RUN:   -disable-llvm-passes -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefixes=CHECK,FAST

// A libm math function that carries #pragma omp declare simd, when called inside
// an OpenMP SIMD loop, gets its call site marked memory(none). This lets the
// loop vectorizer replace it with its VFABI vector variant instead of treating
// it as a dependence-blocking memory access. The tightening is per-call-site:
//   * it fires even under -fmath-errno -fno-builtin (the configuration glibc
//     uses for its libmvec ABI tests), where the function declaration is left
//     with default may-read/write memory effects and FunctionDecl::getBuiltinID
//     returns 0, so the classification must be done by name;
//   * under -fmath-errno the same call outside any SIMD loop is NOT tightened,
//     so ordinary scalar callers keep the function's errno / FP-environment side
//     effects (under fast-math the declaration is globally pure already, so the
//     distinction is not observable there);
//   * functions that write through output pointers (e.g. sincos) are NOT
//     tightened, even inside a SIMD loop.

#pragma omp declare simd notinbranch
double acosh(double);

#pragma omp declare simd notinbranch linear(s) linear(c)
void sincos(double, double *s, double *c);

#define N 128
double a[N], b[N];

// CHECK-LABEL: define{{.*}}@test_simd(
// The acosh call inside the SIMD loop is memory(none) under both configs.
// CHECK: call{{.*}}double @acosh({{.*}}) [[SIMD_ATTR:#[0-9]+]]
void test_simd(void) {
#pragma omp simd
  for (int i = 0; i < N; ++i)
    a[i] = acosh(b[i]);
}

// CHECK-LABEL: define{{.*}}@test_scalar(
// Under -fmath-errno the call outside any SIMD loop is NOT tightened.
// ERRNO: call{{.*}}double @acosh({{.*}}) [[SCALAR_ATTR:#[0-9]+]]
double test_scalar(double x) { return acosh(x); }

// CHECK-LABEL: define{{.*}}@test_sincos_simd(
// sincos has output pointers, so it is NOT tightened even inside a SIMD loop.
// CHECK: call{{.*}}void @sincos({{.*}}ptr{{.*}}ptr{{.*}}) [[SINCOS_ATTR:#[0-9]+]]
void test_sincos_simd(double *s, double *c) {
#pragma omp simd
  for (int i = 0; i < N; ++i)
    sincos(b[i], &s[i], &c[i]);
}

// The in-SIMD-loop math call site is memory(none).
// CHECK: attributes [[SIMD_ATTR]] = { {{.*}}memory(none){{.*}} }
// The sincos call site is never memory(none).
// CHECK-NOT: attributes [[SINCOS_ATTR]] = {{.*}}memory(none)
// Under -fmath-errno the plain scalar call site is not memory(none) either.
// ERRNO-NOT: attributes [[SCALAR_ATTR]] = {{.*}}memory(none)
