// RUN: %clang_cc1 -verify -fopenmp -x c -std=c99 -ast-print %s -o - | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp-simd -x c -std=c99 -ast-print %s -o - | FileCheck %s

// expected-no-diagnostics

int foo(void);

#pragma omp declare variant(foo) match(xxx={}, yyy={ccc})
#pragma omp declare variant(foo) match(xxx={vvv})
#pragma omp declare variant(foo) match(implementation={vendor(ibm)}, implementation={vendor(llvm)})
#pragma omp declare variant(foo) match(implementation={vendor(unknown)})
#pragma omp declare variant(foo) match(implementation={vendor(score(5): ibm)})
int bar(void);

// CHECK:      int foo();
// CHECK-NEXT: #pragma omp declare variant(foo) match(implementation={vendor(score(5):ibm)})
// CHECK-NEXT: #pragma omp declare variant(foo) match(implementation={vendor(unknown)})
// CHECK-NEXT: #pragma omp declare variant(foo) match(implementation={vendor(ibm)})
// CHECK-NEXT: #pragma omp declare variant(foo) match(implementation={vendor(llvm)})
// CHECK-NEXT: int bar();
