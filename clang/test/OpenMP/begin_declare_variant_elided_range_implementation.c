// RUN: %clang_cc1 -triple=x86_64-pc-win32 -verify -fopenmp -x c -std=c99 -fms-extensions -Wno-pragma-pack %s
// RUN: %clang_cc1 -triple=x86_64-pc-win32 -verify -fopenmp-simd -x c -std=c99 -fms-extensions -Wno-pragma-pack %s

// expected-no-diagnostics

#pragma omp begin declare variant match(implementation={vendor(llvm)})
typedef int bar;
void f() {}
#pragma omp end declare variant

#pragma omp begin declare variant match(implementation={vendor(ibm)})
typedef float bar;
void f() {}
#pragma omp end declare variant
