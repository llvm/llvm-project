// RUN: %clang_cc1 -verify -fopenmp -x c -std=c99 -ast-print %s -o - -Wno-openmp-clauses | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp-simd -x c -std=c99 -ast-print %s -o - -Wno-openmp-clauses | FileCheck %s

// expected-no-diagnostics

int foo(void);

#pragma omp declare variant(foo) match(xxx={}, yyy={ccc})
#pragma omp declare variant(foo) match(xxx={vvv})
#pragma omp declare variant(foo) match(implementation={vendor(llvm)})
#pragma omp declare variant(foo) match(implementation={vendor(llvm), xxx})
#pragma omp declare variant(foo) match(implementation={vendor(unknown)})
#pragma omp declare variant(foo) match(implementation={vendor(score(5): ibm, xxx, ibm)})
int bar(void);

// CHECK:      int foo();
// CHECK-NEXT: #pragma omp declare variant(foo) match(implementation={vendor(score(5):ibm, xxx)})
// CHECK-NEXT: #pragma omp declare variant(foo) match(implementation={vendor(unknown)})
// CHECK-NEXT: #pragma omp declare variant(foo) match(implementation={vendor(llvm)})
// CHECK-NEXT: #pragma omp declare variant(foo) match(implementation={vendor(llvm)})
// CHECK-NEXT: int bar();
