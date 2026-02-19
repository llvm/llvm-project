// RUN: %clang_cc1 -DCK1 -verify -fopenmp -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -DCK1 -verify -fopenmp-simd -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// expected-no-diagnostics

void zoo(void) {
  short x[10];
  short *(xp[10]);
  xp[1] = &x[0];
  short **xpp = &xp[0];
  x[1] = 111;

// NOTE: use_device_addr on xpp[1][1] is non-compliant, as the base-pointer
// is xpp[1], which is not a base-language identifier.
#pragma omp target data map(tofrom: xpp[1][1]) //use_device_addr(xpp[1][1])
// FIXME: The assumption that xpp should not be mapped is incorrect.
// The base-pointer of the array-section is xpp[1], not xpp, so the implicit
// clause on xpp, i.e. a zero-length array-section amp, should still be emitted.
#pragma omp target has_device_addr(xpp[1][1])
  {
    xpp[1][1] = 222;
  }
}

// CHECK: @{{.+}} = private unnamed_addr constant [1 x i64] [i64 288]
// SIMD-ONLY0-NOT: @{{.+}} = private unnamed_addr constant [1 x i64] [i64 288]

