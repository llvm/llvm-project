// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=50 -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=50 -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

typedef void *omp_depend_t;

void foo() {}

template <class T>
T tmain(T argc) {
  static T a;
#pragma omp depobj(a) depend(in:argc)
#pragma omp depobj(argc) destroy
#pragma omp depobj(argc) update(inout)
  return argc;
}
// CHECK:      static T a;
// CHECK-NEXT: #pragma omp depobj (a) depend(in : argc){{$}}
// CHECK-NEXT: #pragma omp depobj (argc) destroy{{$}}
// CHECK-NEXT: #pragma omp depobj (argc) update(inout){{$}}
// CHECK:      static void *a;
// CHECK-NEXT: #pragma omp depobj (a) depend(in : argc){{$}}
// CHECK-NEXT: #pragma omp depobj (argc) destroy{{$}}
// CHECK-NEXT: #pragma omp depobj (argc) update(inout){{$}}

int main(int argc, char **argv) {
  static omp_depend_t a;
  omp_depend_t b;
// CHECK: static omp_depend_t a;
// CHECK-NEXT: omp_depend_t b;
#pragma omp depobj(a) depend(out:argc, argv)
#pragma omp depobj(b) destroy
#pragma omp depobj(b) update(mutexinoutset)
// CHECK-NEXT: #pragma omp depobj (a) depend(out : argc,argv)
// CHECK-NEXT: #pragma omp depobj (b) destroy
// CHECK-NEXT: #pragma omp depobj (b) update(mutexinoutset)
  (void)tmain(a), tmain(b);
  return 0;
}

#endif
