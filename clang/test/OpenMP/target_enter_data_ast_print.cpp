// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

template <typename T, int C>
T tmain(T argc, T *argv) {
  T i, j, b, c, d, e, x[20];

  i = argc;
#pragma omp target enter data map(to: i)

#pragma omp target enter data map(to: i) if (target enter data: j > 0)

#pragma omp target enter data map(to: i) if (b)

#pragma omp target enter data map(to: c)

#pragma omp target enter data map(to: c) if(b>e)

#pragma omp target enter data map(alloc: x[0:10], c)

#pragma omp target enter data map(to: c) map(alloc: d)

#pragma omp target enter data map(always,alloc: e)

#pragma omp target enter data nowait map(to: i)

#pragma omp target enter data nowait map(to: i) if (target enter data: j > 0)

#pragma omp target enter data map(to: i) if (b) nowait

#pragma omp target enter data map(to: c) nowait

#pragma omp target enter data map(to: c) nowait if(b>e)

#pragma omp target enter data nowait map(alloc: x[0:10], c)

#pragma omp target enter data nowait map(to: c) map(alloc: d)

#pragma omp target enter data nowait map(always,alloc: e)

#pragma omp target enter data nowait depend(in : argc, argv[i:argc], x[:]) map(to: i)

#pragma omp target enter data nowait map(to: i) if (target enter data: j > 0) depend(in : argc, argv[i:argc], x[:])

#pragma omp target enter data depend(in : argc, argv[i:argc], x[:]) map(to: i) if (b) nowait

#pragma omp target enter data map(to: c) depend(in : argc, argv[i:argc], x[:]) nowait

#pragma omp target enter data map(to: c) nowait if(b>e) depend(in : argc, argv[i:argc], x[:])

#pragma omp target enter data nowait map(alloc: x[0:10], c) depend(in : argc, argv[i:argc], x[:])

#pragma omp target enter data nowait depend(in : argc, argv[i:argc], x[:]) map(to: c) map(alloc: d)

#pragma omp target enter data nowait map(always,alloc: e) depend(in : argc, argv[i:argc], x[:])

  return 0;
}

// CHECK: template <typename T, int C> T tmain(T argc, T *argv) {
// CHECK-NEXT: T i, j, b, c, d, e, x[20];
// CHECK-NEXT: i = argc;
// CHECK-NEXT: #pragma omp target enter data map(to: i)
// CHECK-NEXT: #pragma omp target enter data map(to: i) if(target enter data: j > 0)
// CHECK-NEXT: #pragma omp target enter data map(to: i) if(b)
// CHECK-NEXT: #pragma omp target enter data map(to: c)
// CHECK-NEXT: #pragma omp target enter data map(to: c) if(b > e)
// CHECK-NEXT: #pragma omp target enter data map(alloc: x[0:10],c)
// CHECK-NEXT: #pragma omp target enter data map(to: c) map(alloc: d)
// CHECK-NEXT: #pragma omp target enter data map(always,alloc: e)
// CHECK-NEXT: #pragma omp target enter data nowait map(to: i)
// CHECK-NEXT: #pragma omp target enter data nowait map(to: i) if(target enter data: j > 0)
// CHECK-NEXT: #pragma omp target enter data map(to: i) if(b) nowait
// CHECK-NEXT: #pragma omp target enter data map(to: c) nowait
// CHECK-NEXT: #pragma omp target enter data map(to: c) nowait if(b > e)
// CHECK-NEXT: #pragma omp target enter data nowait map(alloc: x[0:10],c)
// CHECK-NEXT: #pragma omp target enter data nowait map(to: c) map(alloc: d)
// CHECK-NEXT: #pragma omp target enter data nowait map(always,alloc: e)
// CHECK-NEXT: #pragma omp target enter data nowait depend(in : argc,argv[i:argc],x[:]) map(to: i)
// CHECK-NEXT: #pragma omp target enter data nowait map(to: i) if(target enter data: j > 0) depend(in : argc,argv[i:argc],x[:])
// CHECK-NEXT: #pragma omp target enter data depend(in : argc,argv[i:argc],x[:]) map(to: i) if(b) nowait
// CHECK-NEXT: #pragma omp target enter data map(to: c) depend(in : argc,argv[i:argc],x[:]) nowait
// CHECK-NEXT: #pragma omp target enter data map(to: c) nowait if(b > e) depend(in : argc,argv[i:argc],x[:])
// CHECK-NEXT: #pragma omp target enter data nowait map(alloc: x[0:10],c) depend(in : argc,argv[i:argc],x[:])
// CHECK-NEXT: #pragma omp target enter data nowait depend(in : argc,argv[i:argc],x[:]) map(to: c) map(alloc: d)
// CHECK-NEXT: #pragma omp target enter data nowait map(always,alloc: e) depend(in : argc,argv[i:argc],x[:])
// CHECK: template<> int tmain<int, 5>(int argc, int *argv) {
// CHECK-NEXT: int i, j, b, c, d, e, x[20];
// CHECK-NEXT: i = argc;
// CHECK-NEXT: #pragma omp target enter data map(to: i)
// CHECK-NEXT: #pragma omp target enter data map(to: i) if(target enter data: j > 0)
// CHECK-NEXT: #pragma omp target enter data map(to: i) if(b)
// CHECK-NEXT: #pragma omp target enter data map(to: c)
// CHECK-NEXT: #pragma omp target enter data map(to: c) if(b > e)
// CHECK-NEXT: #pragma omp target enter data map(alloc: x[0:10],c)
// CHECK-NEXT: #pragma omp target enter data map(to: c) map(alloc: d)
// CHECK-NEXT: #pragma omp target enter data map(always,alloc: e)
// CHECK-NEXT: #pragma omp target enter data nowait map(to: i)
// CHECK-NEXT: #pragma omp target enter data nowait map(to: i) if(target enter data: j > 0)
// CHECK-NEXT: #pragma omp target enter data map(to: i) if(b) nowait
// CHECK-NEXT: #pragma omp target enter data map(to: c) nowait
// CHECK-NEXT: #pragma omp target enter data map(to: c) nowait if(b > e)
// CHECK-NEXT: #pragma omp target enter data nowait map(alloc: x[0:10],c)
// CHECK-NEXT: #pragma omp target enter data nowait map(to: c) map(alloc: d)
// CHECK-NEXT: #pragma omp target enter data nowait map(always,alloc: e)
// CHECK-NEXT: #pragma omp target enter data nowait depend(in : argc,argv[i:argc],x[:]) map(to: i)
// CHECK-NEXT: #pragma omp target enter data nowait map(to: i) if(target enter data: j > 0) depend(in : argc,argv[i:argc],x[:])
// CHECK-NEXT: #pragma omp target enter data depend(in : argc,argv[i:argc],x[:]) map(to: i) if(b) nowait
// CHECK-NEXT: #pragma omp target enter data map(to: c) depend(in : argc,argv[i:argc],x[:]) nowait
// CHECK-NEXT: #pragma omp target enter data map(to: c) nowait if(b > e) depend(in : argc,argv[i:argc],x[:])
// CHECK-NEXT: #pragma omp target enter data nowait map(alloc: x[0:10],c) depend(in : argc,argv[i:argc],x[:])
// CHECK-NEXT: #pragma omp target enter data nowait depend(in : argc,argv[i:argc],x[:]) map(to: c) map(alloc: d)
// CHECK-NEXT: #pragma omp target enter data nowait map(always,alloc: e) depend(in : argc,argv[i:argc],x[:])
// CHECK: template<> char tmain<char, 1>(char argc, char *argv) {
// CHECK-NEXT: char i, j, b, c, d, e, x[20];
// CHECK-NEXT: i = argc;
// CHECK-NEXT: #pragma omp target enter data map(to: i)
// CHECK-NEXT: #pragma omp target enter data map(to: i) if(target enter data: j > 0)
// CHECK-NEXT: #pragma omp target enter data map(to: i) if(b)
// CHECK-NEXT: #pragma omp target enter data map(to: c)
// CHECK-NEXT: #pragma omp target enter data map(to: c) if(b > e)
// CHECK-NEXT: #pragma omp target enter data map(alloc: x[0:10],c)
// CHECK-NEXT: #pragma omp target enter data map(to: c) map(alloc: d)
// CHECK-NEXT: #pragma omp target enter data map(always,alloc: e)
// CHECK-NEXT: #pragma omp target enter data nowait map(to: i)
// CHECK-NEXT: #pragma omp target enter data nowait map(to: i) if(target enter data: j > 0)
// CHECK-NEXT: #pragma omp target enter data map(to: i) if(b) nowait
// CHECK-NEXT: #pragma omp target enter data map(to: c) nowait
// CHECK-NEXT: #pragma omp target enter data map(to: c) nowait if(b > e)
// CHECK-NEXT: #pragma omp target enter data nowait map(alloc: x[0:10],c)
// CHECK-NEXT: #pragma omp target enter data nowait map(to: c) map(alloc: d)
// CHECK-NEXT: #pragma omp target enter data nowait map(always,alloc: e)
// CHECK-NEXT: #pragma omp target enter data nowait depend(in : argc,argv[i:argc],x[:]) map(to: i)
// CHECK-NEXT: #pragma omp target enter data nowait map(to: i) if(target enter data: j > 0) depend(in : argc,argv[i:argc],x[:])
// CHECK-NEXT: #pragma omp target enter data depend(in : argc,argv[i:argc],x[:]) map(to: i) if(b) nowait
// CHECK-NEXT: #pragma omp target enter data map(to: c) depend(in : argc,argv[i:argc],x[:]) nowait
// CHECK-NEXT: #pragma omp target enter data map(to: c) nowait if(b > e) depend(in : argc,argv[i:argc],x[:])
// CHECK-NEXT: #pragma omp target enter data nowait map(alloc: x[0:10],c) depend(in : argc,argv[i:argc],x[:])
// CHECK-NEXT: #pragma omp target enter data nowait depend(in : argc,argv[i:argc],x[:]) map(to: c) map(alloc: d)
// CHECK-NEXT: #pragma omp target enter data nowait map(always,alloc: e) depend(in : argc,argv[i:argc],x[:])

int main (int argc, char **argv) {
  int b = argc, i, c, d, e, f, g, x[20];
  static int a;
// CHECK: static int a;

#pragma omp target enter data map(to: a)
// CHECK:      #pragma omp target enter data map(to: a)
  a=2;
// CHECK-NEXT: a = 2;
#pragma omp target enter data map(to: a) if (target enter data: b)
// CHECK: #pragma omp target enter data map(to: a) if(target enter data: b)

#pragma omp target enter data map(to: a) if (b > g)
// CHECK: #pragma omp target enter data map(to: a) if(b > g)

#pragma omp target enter data map(to: c)
// CHECK-NEXT: #pragma omp target enter data map(to: c)

#pragma omp target enter data map(alloc: c) if(b>g)
// CHECK-NEXT: #pragma omp target enter data map(alloc: c) if(b > g)

#pragma omp target enter data map(to: x[0:10], c)
// CHECK-NEXT: #pragma omp target enter data map(to: x[0:10],c)

#pragma omp target enter data map(to: c) map(alloc: d)
// CHECK-NEXT: #pragma omp target enter data map(to: c) map(alloc: d)

#pragma omp target enter data map(always,alloc: e)
// CHECK-NEXT: #pragma omp target enter data map(always,alloc: e)

#pragma omp target enter data nowait map(to: a)
// CHECK:      #pragma omp target enter data nowait map(to: a)

#pragma omp target enter data nowait map(to: a) if (target enter data: b)
// CHECK: #pragma omp target enter data nowait map(to: a) if(target enter data: b)

#pragma omp target enter data map(to: a) if (b > g) nowait
// CHECK: #pragma omp target enter data map(to: a) if(b > g) nowait

#pragma omp target enter data map(to: c) nowait
// CHECK-NEXT: #pragma omp target enter data map(to: c) nowait

#pragma omp target enter data map(alloc: c) nowait if(b>g)
// CHECK-NEXT: #pragma omp target enter data map(alloc: c) nowait if(b > g)

#pragma omp target enter data nowait map(to: x[0:10], c)
// CHECK-NEXT: #pragma omp target enter data nowait map(to: x[0:10],c)

#pragma omp target enter data nowait map(to: c) map(alloc: d)
// CHECK-NEXT: #pragma omp target enter data nowait map(to: c) map(alloc: d)

#pragma omp target enter data nowait map(always,alloc: e)
// CHECK-NEXT: #pragma omp target enter data nowait map(always,alloc: e)

#pragma omp target enter data depend(in : argc, argv[i:argc], x[:]) nowait map(to: a)
// CHECK:      #pragma omp target enter data depend(in : argc,argv[i:argc],x[:]) nowait map(to: a)

#pragma omp target enter data nowait depend(in : argc, argv[i:argc], x[:]) map(to: a) if (target enter data: b)
// CHECK: #pragma omp target enter data nowait depend(in : argc,argv[i:argc],x[:]) map(to: a) if(target enter data: b)

#pragma omp target enter data map(to: a) depend(in : argc, argv[i:argc], x[:]) if (b > g) nowait
// CHECK: #pragma omp target enter data map(to: a) depend(in : argc,argv[i:argc],x[:]) if(b > g) nowait

#pragma omp target enter data map(to: c) nowait depend(in : argc, argv[i:argc], x[:])
// CHECK-NEXT: #pragma omp target enter data map(to: c) nowait depend(in : argc,argv[i:argc],x[:])

#pragma omp target enter data depend(in : argc, argv[i:argc], x[:]) map(alloc: c) nowait if(b>g)
// CHECK-NEXT: #pragma omp target enter data depend(in : argc,argv[i:argc],x[:]) map(alloc: c) nowait if(b > g)

#pragma omp target enter data nowait map(to: x[0:10], c) depend(in : argc, argv[i:argc], x[:])
// CHECK-NEXT: #pragma omp target enter data nowait map(to: x[0:10],c) depend(in : argc,argv[i:argc],x[:])

#pragma omp target enter data nowait map(to: c) depend(in : argc, argv[i:argc], x[:]) map(alloc: d)
// CHECK-NEXT: #pragma omp target enter data nowait map(to: c) depend(in : argc,argv[i:argc],x[:]) map(alloc: d)

#pragma omp target enter data nowait map(always,alloc: e) depend(in : argc, argv[i:argc], x[:])
// CHECK-NEXT: #pragma omp target enter data nowait map(always,alloc: e) depend(in : argc,argv[i:argc],x[:])

  return tmain<int, 5>(argc, &argc) + tmain<char, 1>(argv[0][0], argv[0]);
}

#endif
