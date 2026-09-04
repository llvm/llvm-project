// Optimized split IR at -O1; split + `-fopenmp-simd` syntax-only; -g debug-info smoke.
//
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 -O1 -emit-llvm -DTEST_BODY %s -o - | FileCheck %s --check-prefix=O1
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp-simd -fopenmp-version=60 -fsyntax-only -verify -DTEST_SIMD %s
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 -O0 -emit-llvm -debug-info-kind=limited -DTEST_BODY %s -o - | FileCheck %s --check-prefix=DBG

extern "C" void body(int);

#if defined(TEST_SIMD)
// expected-no-diagnostics
void simd_ok(int n) {
#pragma omp split counts(2, omp_fill)
  for (int i = 0; i < n; ++i)
    body(i);
}
#endif

#if defined(TEST_BODY)
// O1-LABEL: define {{.*}} @_Z4testi
// O1: .split.iv
// DBG-LABEL: define {{.*}} @_Z4testi
// DBG: .split.iv
// DBG: !dbg
void test(int n) {
#pragma omp split counts(2, omp_fill)
  for (int i = 0; i < n; ++i)
    body(i);
}
#endif

