// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=51 -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=51 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=51 -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=51 -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=51 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=51 -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

void foo() {}

template <class T>
T tmain(T argc) {
  static T a;
#pragma omp taskwait
#pragma omp taskwait depend(in:a, argc)
  return a + argc;
}

template <class T>
T ndmain(T argc) {
  static T a;
#pragma omp taskwait
#pragma omp taskwait depend(in:a, argc) nowait
  return a + argc;
}

// CHECK:      static T a;
// CHECK-NEXT: #pragma omp taskwait{{$}}
// CHECK-NEXT: #pragma omp taskwait depend(in : a,argc){{$}}
// CHECK:      static int a;
// CHECK-NEXT: #pragma omp taskwait
// CHECK-NEXT: #pragma omp taskwait depend(in : a,argc){{$}}
// CHECK:      static char a;
// CHECK-NEXT: #pragma omp taskwait
// CHECK-NEXT: #pragma omp taskwait depend(in : a,argc){{$}}

// CHECK:      static T a;
// CHECK-NEXT: #pragma omp taskwait{{$}}
// CHECK-NEXT: #pragma omp taskwait depend(in : a,argc) nowait{{$}}
// CHECK:      static int a;
// CHECK-NEXT: #pragma omp taskwait
// CHECK-NEXT: #pragma omp taskwait depend(in : a,argc) nowait{{$}}
// CHECK:      static char a;
// CHECK-NEXT: #pragma omp taskwait
// CHECK-NEXT: #pragma omp taskwait depend(in : a,argc) nowait{{$}}

int main(int argc, char **argv) {
  static int a;
// CHECK: static int a;
#pragma omp taskwait
#pragma omp taskwait depend(out:a, argc)
  // CHECK-NEXT: #pragma omp taskwait
  // CHECK-NEXT: #pragma omp taskwait depend(out : a,argc)
  return tmain(argc) + tmain(argv[0][0]) + a + ndmain(argc) + ndmain(argv[0][0]);
}

#endif
