// RUN: %clang_cc1 -verify -fopenmp -x c -std=c99 -ast-print %s -o - | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp-simd -x c -std=c99 -ast-print %s -o - | FileCheck %s

// expected-no-diagnostics

int foo(void);

#pragma omp declare variant(foo) match(xxx={}, yyy={ccc})
#pragma omp declare variant(foo) match(xxx={vvv})
int bar(void);

// CHECK:      int foo();
// CHECK-NEXT: #pragma omp declare variant(foo) match(unknown={})
// CHECK-NEXT: #pragma omp declare variant(foo) match(unknown={})
// CHECK-NEXT: #pragma omp declare variant(foo) match(unknown={})
// CHECK-NEXT: int bar();
