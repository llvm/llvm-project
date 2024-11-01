// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=45 -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=45 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=45 -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=52 -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=52 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=52 -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp-simd -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s

// expected-no-diagnostics

#ifndef HEADER
#define HEADER

template <typename T, int C>
T tmain(T argc, T *argv) {
  T i, j, b, c, d, e, x[20];

  i = argc;
#pragma omp target exit data map(from: i)

#pragma omp target exit data map(from: i) if (target exit data: j > 0)

#pragma omp target exit data map(from: i) if (b)

#pragma omp target exit data map(from: c)

#pragma omp target exit data map(from: c) if(b>e)

#pragma omp target exit data map(release: x[0:10], c)

#pragma omp target exit data map(delete: x[0:10])

#pragma omp target exit data map(always, delete: x[0:10])

#pragma omp target exit data map(from: c) map(release: d)

#pragma omp target exit data map(always,release: e)

#pragma omp target exit data nowait map(from: i)

#pragma omp target exit data nowait map(from: i) if (target exit data: j > 0)

#pragma omp target exit data map(from: i) if (b) nowait

#pragma omp target exit data map(from: c) nowait

#pragma omp target exit data map(from: c) nowait if(b>e)

#pragma omp target exit data nowait map(release: x[0:10], c)

#pragma omp target exit data nowait map(from: c) map(release: d)

#pragma omp target exit data nowait map(always,release: e)

#pragma omp target exit data depend(in : argc, argv[i:argc], x[:]) nowait map(from: i)

#pragma omp target exit data nowait depend(in : argc, argv[i:argc], x[:]) map(from: i) if (target exit data: j > 0)

#pragma omp target exit data map(from: i) depend(in : argc, argv[i:argc], x[:]) if (b) nowait

#pragma omp target exit data map(from: c) depend(in : argc, argv[i:argc], x[:]) nowait

#pragma omp target exit data map(from: c) depend(in : argc, argv[i:argc], x[:]) nowait if(b>e)

#pragma omp target exit data nowait map(release: x[0:10], c) depend(in : argc, argv[i:argc], x[:])

#pragma omp target exit data nowait map(from: c) depend(in : argc, argv[i:argc], x[:]) map(release: d)

#pragma omp target exit data depend(in : argc, argv[i:argc], x[:]) nowait map(always,release: e)

  return 0;
}

// CHECK: template <typename T, int C> T tmain(T argc, T *argv) {
// CHECK-NEXT: T i, j, b, c, d, e, x[20];
// CHECK-NEXT: i = argc;
// CHECK-NEXT: #pragma omp target exit data map(from: i){{$}}
// CHECK-NEXT: #pragma omp target exit data map(from: i) if(target exit data: j > 0)
// CHECK-NEXT: #pragma omp target exit data map(from: i) if(b)
// CHECK-NEXT: #pragma omp target exit data map(from: c)
// CHECK-NEXT: #pragma omp target exit data map(from: c) if(b > e)
// CHECK-NEXT: #pragma omp target exit data map(release: x[0:10],c)
// CHECK-NEXT: #pragma omp target exit data map(delete: x[0:10])
// CHECK-NEXT: #pragma omp target exit data map(always,delete: x[0:10])
// CHECK-NEXT: #pragma omp target exit data map(from: c) map(release: d)
// CHECK-NEXT: #pragma omp target exit data map(always,release: e)
// CHECK-NEXT: #pragma omp target exit data nowait map(from: i)
// CHECK-NEXT: #pragma omp target exit data nowait map(from: i) if(target exit data: j > 0)
// CHECK-NEXT: #pragma omp target exit data map(from: i) if(b) nowait
// CHECK-NEXT: #pragma omp target exit data map(from: c) nowait
// CHECK-NEXT: #pragma omp target exit data map(from: c) nowait if(b > e)
// CHECK-NEXT: #pragma omp target exit data nowait map(release: x[0:10],c)
// CHECK-NEXT: #pragma omp target exit data nowait map(from: c) map(release: d)
// CHECK-NEXT: #pragma omp target exit data nowait map(always,release: e)
// CHECK-NEXT: #pragma omp target exit data depend(in : argc,argv[i:argc],x[:]) nowait map(from: i)
// CHECK-NEXT: #pragma omp target exit data nowait depend(in : argc,argv[i:argc],x[:]) map(from: i) if(target exit data: j > 0)
// CHECK-NEXT: #pragma omp target exit data map(from: i) depend(in : argc,argv[i:argc],x[:]) if(b) nowait
// CHECK-NEXT: #pragma omp target exit data map(from: c) depend(in : argc,argv[i:argc],x[:]) nowait
// CHECK-NEXT: #pragma omp target exit data map(from: c) depend(in : argc,argv[i:argc],x[:]) nowait if(b > e)
// CHECK-NEXT: #pragma omp target exit data nowait map(release: x[0:10],c) depend(in : argc,argv[i:argc],x[:])
// CHECK-NEXT: #pragma omp target exit data nowait map(from: c) depend(in : argc,argv[i:argc],x[:]) map(release: d)
// CHECK-NEXT: #pragma omp target exit data depend(in : argc,argv[i:argc],x[:]) nowait map(always,release: e)
// CHECK: template<> int tmain<int, 5>(int argc, int *argv) {
// CHECK-NEXT: int i, j, b, c, d, e, x[20];
// CHECK-NEXT: i = argc;
// CHECK-NEXT: #pragma omp target exit data map(from: i)
// CHECK-NEXT: #pragma omp target exit data map(from: i) if(target exit data: j > 0)
// CHECK-NEXT: #pragma omp target exit data map(from: i) if(b)
// CHECK-NEXT: #pragma omp target exit data map(from: c)
// CHECK-NEXT: #pragma omp target exit data map(from: c) if(b > e)
// CHECK-NEXT: #pragma omp target exit data map(release: x[0:10],c)
// CHECK-NEXT: #pragma omp target exit data map(delete: x[0:10])
// CHECK-NEXT: #pragma omp target exit data map(always,delete: x[0:10])
// CHECK-NEXT: #pragma omp target exit data map(from: c) map(release: d)
// CHECK-NEXT: #pragma omp target exit data map(always,release: e)
// CHECK-NEXT: #pragma omp target exit data nowait map(from: i)
// CHECK-NEXT: #pragma omp target exit data nowait map(from: i) if(target exit data: j > 0)
// CHECK-NEXT: #pragma omp target exit data map(from: i) if(b) nowait
// CHECK-NEXT: #pragma omp target exit data map(from: c) nowait
// CHECK-NEXT: #pragma omp target exit data map(from: c) nowait if(b > e)
// CHECK-NEXT: #pragma omp target exit data nowait map(release: x[0:10],c)
// CHECK-NEXT: #pragma omp target exit data nowait map(from: c) map(release: d)
// CHECK-NEXT: #pragma omp target exit data nowait map(always,release: e)
// CHECK-NEXT: #pragma omp target exit data depend(in : argc,argv[i:argc],x[:]) nowait map(from: i)
// CHECK-NEXT: #pragma omp target exit data nowait depend(in : argc,argv[i:argc],x[:]) map(from: i) if(target exit data: j > 0)
// CHECK-NEXT: #pragma omp target exit data map(from: i) depend(in : argc,argv[i:argc],x[:]) if(b) nowait
// CHECK-NEXT: #pragma omp target exit data map(from: c) depend(in : argc,argv[i:argc],x[:]) nowait
// CHECK-NEXT: #pragma omp target exit data map(from: c) depend(in : argc,argv[i:argc],x[:]) nowait if(b > e)
// CHECK-NEXT: #pragma omp target exit data nowait map(release: x[0:10],c) depend(in : argc,argv[i:argc],x[:])
// CHECK-NEXT: #pragma omp target exit data nowait map(from: c) depend(in : argc,argv[i:argc],x[:]) map(release: d)
// CHECK-NEXT: #pragma omp target exit data depend(in : argc,argv[i:argc],x[:]) nowait map(always,release: e)
// CHECK: template<> char tmain<char, 1>(char argc, char *argv) {
// CHECK-NEXT: char i, j, b, c, d, e, x[20];
// CHECK-NEXT: i = argc;
// CHECK-NEXT: #pragma omp target exit data map(from: i)
// CHECK-NEXT: #pragma omp target exit data map(from: i) if(target exit data: j > 0)
// CHECK-NEXT: #pragma omp target exit data map(from: i) if(b)
// CHECK-NEXT: #pragma omp target exit data map(from: c)
// CHECK-NEXT: #pragma omp target exit data map(from: c) if(b > e)
// CHECK-NEXT: #pragma omp target exit data map(release: x[0:10],c)
// CHECK-NEXT: #pragma omp target exit data map(delete: x[0:10])
// CHECK-NEXT: #pragma omp target exit data map(always,delete: x[0:10])
// CHECK-NEXT: #pragma omp target exit data map(from: c) map(release: d)
// CHECK-NEXT: #pragma omp target exit data map(always,release: e)
// CHECK-NEXT: #pragma omp target exit data nowait map(from: i)
// CHECK-NEXT: #pragma omp target exit data nowait map(from: i) if(target exit data: j > 0)
// CHECK-NEXT: #pragma omp target exit data map(from: i) if(b) nowait
// CHECK-NEXT: #pragma omp target exit data map(from: c) nowait
// CHECK-NEXT: #pragma omp target exit data map(from: c) nowait if(b > e)
// CHECK-NEXT: #pragma omp target exit data nowait map(release: x[0:10],c)
// CHECK-NEXT: #pragma omp target exit data nowait map(from: c) map(release: d)
// CHECK-NEXT: #pragma omp target exit data nowait map(always,release: e)
// CHECK-NEXT: #pragma omp target exit data depend(in : argc,argv[i:argc],x[:]) nowait map(from: i)
// CHECK-NEXT: #pragma omp target exit data nowait depend(in : argc,argv[i:argc],x[:]) map(from: i) if(target exit data: j > 0)
// CHECK-NEXT: #pragma omp target exit data map(from: i) depend(in : argc,argv[i:argc],x[:]) if(b) nowait
// CHECK-NEXT: #pragma omp target exit data map(from: c) depend(in : argc,argv[i:argc],x[:]) nowait
// CHECK-NEXT: #pragma omp target exit data map(from: c) depend(in : argc,argv[i:argc],x[:]) nowait if(b > e)
// CHECK-NEXT: #pragma omp target exit data nowait map(release: x[0:10],c) depend(in : argc,argv[i:argc],x[:])
// CHECK-NEXT: #pragma omp target exit data nowait map(from: c) depend(in : argc,argv[i:argc],x[:]) map(release: d)
// CHECK-NEXT: #pragma omp target exit data depend(in : argc,argv[i:argc],x[:]) nowait map(always,release: e)

int main (int argc, char **argv) {
  int b = argc, i, c, d, e, f, g, x[20];
  static int a;
// CHECK: static int a;

#pragma omp target exit data map(from: a)
// CHECK:      #pragma omp target exit data map(from: a)
  a=2;
// CHECK-NEXT: a = 2;
#pragma omp target exit data map(from: a) if (target exit data: b)
// CHECK: #pragma omp target exit data map(from: a) if(target exit data: b)

#pragma omp target exit data map(from: a) if (b > g)
// CHECK: #pragma omp target exit data map(from: a) if(b > g)

#pragma omp target exit data map(from: c)
// CHECK-NEXT: #pragma omp target exit data map(from: c)

#pragma omp target exit data map(release: c) if(b>g)
// CHECK-NEXT: #pragma omp target exit data map(release: c) if(b > g)

#pragma omp target exit data map(from: x[0:10], c)
// CHECK-NEXT: #pragma omp target exit data map(from: x[0:10],c)

#pragma omp target exit data map(delete: x[0:10])
// CHECK-NEXT: #pragma omp target exit data map(delete: x[0:10])

#pragma omp target exit data map(always, delete: x[0:10])
// CHECK-NEXT: #pragma omp target exit data map(always,delete: x[0:10])

#pragma omp target exit data map(from: c) map(release: d)
// CHECK-NEXT: #pragma omp target exit data map(from: c) map(release: d)

#pragma omp target exit data map(always,release: e)
// CHECK-NEXT: #pragma omp target exit data map(always,release: e)

#pragma omp target exit data nowait map(from: a)
// CHECK:      #pragma omp target exit data nowait map(from: a)

#pragma omp target exit data nowait map(from: a) if (target exit data: b)
// CHECK: #pragma omp target exit data nowait map(from: a) if(target exit data: b)

#pragma omp target exit data map(from: a) if (b > g) nowait
// CHECK: #pragma omp target exit data map(from: a) if(b > g) nowait

#pragma omp target exit data map(from: c) nowait
// CHECK-NEXT: #pragma omp target exit data map(from: c) nowait

#pragma omp target exit data map(release: c) nowait if(b>g)
// CHECK-NEXT: #pragma omp target exit data map(release: c) nowait if(b > g)

#pragma omp target exit data nowait map(from: x[0:10], c)
// CHECK-NEXT: #pragma omp target exit data nowait map(from: x[0:10],c)

#pragma omp target exit data nowait map(from: c) map(release: d)
// CHECK-NEXT: #pragma omp target exit data nowait map(from: c) map(release: d)

#pragma omp target exit data nowait map(always,release: e)
// CHECK-NEXT: #pragma omp target exit data nowait map(always,release: e)

#pragma omp target exit data nowait depend(in : argc,argv[i:argc],x[:]) map(from: a)
// CHECK:      #pragma omp target exit data nowait depend(in : argc,argv[i:argc],x[:]) map(from: a)

#pragma omp target exit data nowait map(from: a) depend(in : argc,argv[i:argc],x[:]) if (target exit data: b)
// CHECK: #pragma omp target exit data nowait map(from: a) depend(in : argc,argv[i:argc],x[:]) if(target exit data: b)

#pragma omp target exit data map(from: a) if (b > g) nowait depend(in : argc,argv[i:argc],x[:])
// CHECK: #pragma omp target exit data map(from: a) if(b > g) nowait depend(in : argc,argv[i:argc],x[:])

#pragma omp target exit data map(from: c) depend(in : argc,argv[i:argc],x[:]) nowait
// CHECK-NEXT: #pragma omp target exit data map(from: c) depend(in : argc,argv[i:argc],x[:]) nowait

#pragma omp target exit data depend(in : argc,argv[i:argc],x[:]) map(release: c) nowait if(b>g)
// CHECK-NEXT: #pragma omp target exit data depend(in : argc,argv[i:argc],x[:]) map(release: c) nowait if(b > g)

#pragma omp target exit data nowait map(from: x[0:10], c) depend(in : argc,argv[i:argc],x[:])
// CHECK-NEXT: #pragma omp target exit data nowait map(from: x[0:10],c) depend(in : argc,argv[i:argc],x[:])

#pragma omp target exit data nowait map(from: c) depend(in : argc,argv[i:argc],x[:]) map(release: d)
// CHECK-NEXT: #pragma omp target exit data nowait map(from: c) depend(in : argc,argv[i:argc],x[:]) map(release: d)

#pragma omp target exit data nowait depend(in : argc,argv[i:argc],x[:]) map(always,release: e)
// CHECK-NEXT: #pragma omp target exit data nowait depend(in : argc,argv[i:argc],x[:]) map(always,release: e)

  return tmain<int, 5>(argc, &argc) + tmain<char, 1>(argv[0][0], argv[0]);
}

#endif
