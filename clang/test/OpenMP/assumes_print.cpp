// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=60 -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=60 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=60 -std=c++11 -include-pch %t -verify %s -ast-print | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=60 -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=60 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=60 -std=c++11 -include-pch %t -verify %s -ast-print | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

void foo() {
}

#pragma omp assumes no_openmp_routines
#pragma omp assumes no_openmp_constructs

namespace inner {
  #pragma omp assumes no_openmp
} // namespace inner

#pragma omp begin assumes ext_range_bar_only

#pragma omp begin assumes ext_range_bar_only_2

void bar() {
}

#pragma omp end assumes
#pragma omp end assumes

#pragma omp begin assumes ext_not_seen
#pragma omp end assumes

#pragma omp begin assumes ext_1234
void baz() {
}
#pragma omp end assumes

// CHECK{LITERAL}: void foo() [[omp::assume("omp_no_openmp_routines")]] [[omp::assume("omp_no_openmp_constructs")]] [[omp::assume("omp_no_openmp")]]
// CHECK{LITERAL}: [[omp::assume("ompx_range_bar_only")]] [[omp::assume("ompx_range_bar_only_2")]] [[omp::assume("omp_no_openmp_routines")]] [[omp::assume("omp_no_openmp_constructs")]] [[omp::assume("omp_no_openmp")]] void bar()
// CHECK{LITERAL}: [[omp::assume("ompx_1234")]] [[omp::assume("omp_no_openmp_routines")]] [[omp::assume("omp_no_openmp_constructs")]] [[omp::assume("omp_no_openmp")]] void baz()

#endif
