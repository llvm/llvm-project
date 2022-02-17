// RUN: %clang_cc1 -verify -fopenmp -x c -std=c99 -ast-print %s -o - -Wno-openmp-clauses | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp-simd -x c -std=c99 -ast-print %s -o - -Wno-openmp-clauses | FileCheck %s

// expected-no-diagnostics

int foo(void);

#pragma omp declare variant(foo) match(construct={target})
#pragma omp declare variant(foo) match(construct={teams})
#pragma omp declare variant(foo) match(construct={parallel})
#pragma omp declare variant(foo) match(construct={for})
#pragma omp declare variant(foo) match(construct={simd})
#pragma omp declare variant(foo) match(construct={target,teams,parallel,for,simd})
#pragma omp declare variant(foo) match(xxx={}, yyy={ccc})
#pragma omp declare variant(foo) match(xxx={vvv})
#pragma omp declare variant(foo) match(implementation={vendor(score(0):llvm)}, device={kind(fpga)})
#pragma omp declare variant(foo) match(implementation={vendor(llvm), xxx})
#pragma omp declare variant(foo) match(implementation={vendor(unknown)}, device={kind(gpu)})
#pragma omp declare variant(foo) match(implementation={vendor(score(5): ibm, xxx, ibm)}, device={kind(cpu, nohost)})
#pragma omp declare variant(foo) match(device={kind(host)})
#pragma omp declare variant(foo) match(device={kind(nohost), xxx})
#pragma omp declare variant(foo) match(implementation={extension(match_all)})
#pragma omp declare variant(foo) match(implementation={extension(match_any)})
#pragma omp declare variant(foo) match(implementation={extension(match_none)})
int bar(void);

// CHECK:      int foo(void);
// CHECK-NEXT: #pragma omp declare variant(foo) match(implementation={extension(match_none)})
// CHECK-NEXT: #pragma omp declare variant(foo) match(implementation={extension(match_any)})
// CHECK-NEXT: #pragma omp declare variant(foo) match(implementation={extension(match_all)})
// CHECK-NEXT: #pragma omp declare variant(foo) match(device={kind(nohost)})
// CHECK-NEXT: #pragma omp declare variant(foo) match(device={kind(host)})
// CHECK-NEXT: #pragma omp declare variant(foo) match(implementation={vendor(score(5): ibm)}, device={kind(cpu, nohost)})
// CHECK-NEXT: #pragma omp declare variant(foo) match(implementation={vendor(unknown)}, device={kind(gpu)})
// CHECK-NEXT: #pragma omp declare variant(foo) match(implementation={vendor(llvm)})
// CHECK-NEXT: #pragma omp declare variant(foo) match(implementation={vendor(score(0): llvm)}, device={kind(fpga)})
// CHECK-NEXT: #pragma omp declare variant(foo) match(construct={target, teams, parallel, for, simd})
// CHECK-NEXT: #pragma omp declare variant(foo) match(construct={simd})
// CHECK-NEXT: #pragma omp declare variant(foo) match(construct={for})
// CHECK-NEXT: #pragma omp declare variant(foo) match(construct={parallel})
// CHECK-NEXT: #pragma omp declare variant(foo) match(construct={teams})
// CHECK-NEXT: #pragma omp declare variant(foo) match(construct={target})
// CHECK-NEXT: int bar(void);
