// REQUIRES: aarch64-registered-target
// -fopemp and -fopenmp-simd behavior are expected to be the same

// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +sve \
// RUN:  -fopenmp -x c -emit-llvm %s -o - -femit-all-decls | FileCheck %s

// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +sve \
// RUN:  -fopenmp-simd -x c -emit-llvm %s -o - -femit-all-decls | FileCheck %s

#pragma omp declare simd
#pragma omp declare simd notinbranch
#pragma omp declare simd simdlen(2)
#pragma omp declare simd simdlen(4)
#pragma omp declare simd simdlen(5) // not a multiple of 128-bits
#pragma omp declare simd simdlen(6)
#pragma omp declare simd simdlen(8)
#pragma omp declare simd simdlen(32)
#pragma omp declare simd simdlen(34) // requires more than 2048 bits
double foo(float x);

// CHECK-DAG: "_ZGVsM2v_foo" "_ZGVsM32v_foo" "_ZGVsM4v_foo" "_ZGVsM6v_foo" "_ZGVsM8v_foo" "_ZGVsMxv_foo"
// CHECK-NOT: _ZGVsN
// CHECK-NOT: _ZGVsM5v_foo
// CHECK-NOT: _ZGVsM34v_foo
// CHECK-NOT: foo

void foo_loop(double *x, float *y, int N) {
  for (int i = 0; i < N; ++i) {
    x[i] = foo(y[i]);
  }
}

  // test integers

#pragma omp declare simd notinbranch
char a01(int x);
// CHECK-DAG: _ZGVsMxv_a01
// CHECK-NOT: a01

static int *in;
static char *out;
void do_something() {
  *out = a01(*in);
}
