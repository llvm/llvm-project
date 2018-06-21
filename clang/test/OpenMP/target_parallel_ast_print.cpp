// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=45 -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=45 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=45 -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

void foo() {}

template <class T>
struct S {
  operator T() {return T();}
  static T TS;
  #pragma omp threadprivate(TS)
};

// CHECK:      template <class T> struct S {
// CHECK:        static T TS;
// CHECK-NEXT:   #pragma omp threadprivate(S::TS)
// CHECK:      };
// CHECK:      template<> struct S<int> {
// CHECK:        static int TS;
// CHECK-NEXT:   #pragma omp threadprivate(S<int>::TS)
// CHECK-NEXT: }
// CHECK:      template<> struct S<char> {
// CHECK:        static char TS;
// CHECK-NEXT:   #pragma omp threadprivate(S<char>::TS)
// CHECK-NEXT: }

template <typename T, int C>
T tmain(T argc, T *argv) {
  T b = argc, c, d, e, f, g;
  static T h;
  S<T> s;
  T arr[C][10], arr1[C];
  T i, j, a[20];
#pragma omp target parallel
  h=2;
#pragma omp target parallel default(none), private(argc,b) firstprivate(argv) shared (d) if (parallel:argc > 0) num_threads(C) proc_bind(master) reduction(+:c, arr1[argc]) reduction(max:e, arr[:C][0:10])
  foo();
#pragma omp target parallel if (C) num_threads(s) proc_bind(close) reduction(^:e, f, arr[0:C][:argc]) reduction(&& : g)
  foo();
#pragma omp target parallel if (target:argc > 0)
  foo();
#pragma omp target parallel if (parallel:argc > 0)
  foo();
#pragma omp target parallel if (C)
  foo();
#pragma omp target parallel map(i)
  foo();
#pragma omp target parallel map(a[0:10], i)
  foo();
#pragma omp target parallel map(to: i) map(from: j)
  foo();
#pragma omp target parallel map(always,alloc: i)
  foo();
#pragma omp target parallel nowait
  foo();
#pragma omp target parallel depend(in : argc, argv[i:argc], a[:])
  foo();
#pragma omp target parallel defaultmap(tofrom: scalar)
  foo();
  return 0;
}

// CHECK: template <typename T, int C> T tmain(T argc, T *argv) {
// CHECK-NEXT: T b = argc, c, d, e, f, g;
// CHECK-NEXT: static T h;
// CHECK-NEXT: S<T> s;
// CHECK-NEXT: T arr[C][10], arr1[C];
// CHECK-NEXT: T i, j, a[20]
// CHECK-NEXT: #pragma omp target parallel
// CHECK-NEXT: h = 2;
// CHECK-NEXT: #pragma omp target parallel default(none) private(argc,b) firstprivate(argv) shared(d) if(parallel: argc > 0) num_threads(C) proc_bind(master) reduction(+: c,arr1[argc]) reduction(max: e,arr[:C][0:10])
// CHECK-NEXT: foo()
// CHECK-NEXT: #pragma omp target parallel if(C) num_threads(s) proc_bind(close) reduction(^: e,f,arr[0:C][:argc]) reduction(&&: g)
// CHECK-NEXT: foo()
// CHECK-NEXT: #pragma omp target parallel if(target: argc > 0)
// CHECK-NEXT: foo()
// CHECK-NEXT: #pragma omp target parallel if(parallel: argc > 0)
// CHECK-NEXT: foo()
// CHECK-NEXT: #pragma omp target parallel if(C)
// CHECK-NEXT: foo()
// CHECK-NEXT: #pragma omp target parallel map(tofrom: i)
// CHECK-NEXT: foo()
// CHECK-NEXT: #pragma omp target parallel map(tofrom: a[0:10],i)
// CHECK-NEXT: foo()
// CHECK-NEXT: #pragma omp target parallel map(to: i) map(from: j)
// CHECK-NEXT: foo()
// CHECK-NEXT: #pragma omp target parallel map(always,alloc: i)
// CHECK-NEXT: foo()
// CHECK-NEXT: #pragma omp target parallel nowait
// CHECK-NEXT: foo()
// CHECK-NEXT: #pragma omp target parallel depend(in : argc,argv[i:argc],a[:])
// CHECK-NEXT: foo()
// CHECK-NEXT: #pragma omp target parallel defaultmap(tofrom: scalar)
// CHECK-NEXT: foo()
// CHECK: template<> int tmain<int, 5>(int argc, int *argv) {
// CHECK-NEXT: int b = argc, c, d, e, f, g;
// CHECK-NEXT: static int h;
// CHECK-NEXT: S<int> s;
// CHECK-NEXT: int arr[5][10], arr1[5];
// CHECK-NEXT: int i, j, a[20]
// CHECK-NEXT: #pragma omp target parallel
// CHECK-NEXT: h = 2;
// CHECK-NEXT: #pragma omp target parallel default(none) private(argc,b) firstprivate(argv) shared(d) if(parallel: argc > 0) num_threads(5) proc_bind(master) reduction(+: c,arr1[argc]) reduction(max: e,arr[:5][0:10])
// CHECK-NEXT: foo()
// CHECK-NEXT: #pragma omp target parallel if(5) num_threads(s) proc_bind(close) reduction(^: e,f,arr[0:5][:argc]) reduction(&&: g)
// CHECK-NEXT: foo()
// CHECK-NEXT: #pragma omp target parallel if(target: argc > 0)
// CHECK-NEXT: foo()
// CHECK-NEXT: #pragma omp target parallel if(parallel: argc > 0)
// CHECK-NEXT: foo()
// CHECK-NEXT: #pragma omp target parallel if(5)
// CHECK-NEXT: foo()
// CHECK-NEXT: #pragma omp target parallel map(tofrom: i)
// CHECK-NEXT: foo()
// CHECK-NEXT: #pragma omp target parallel map(tofrom: a[0:10],i)
// CHECK-NEXT: foo()
// CHECK-NEXT: #pragma omp target parallel map(to: i) map(from: j)
// CHECK-NEXT: foo()
// CHECK-NEXT: #pragma omp target parallel map(always,alloc: i)
// CHECK-NEXT: foo()
// CHECK-NEXT: #pragma omp target parallel nowait
// CHECK-NEXT: foo()
// CHECK-NEXT: #pragma omp target parallel depend(in : argc,argv[i:argc],a[:])
// CHECK-NEXT: foo()
// CHECK-NEXT: #pragma omp target parallel defaultmap(tofrom: scalar)
// CHECK-NEXT: foo()
// CHECK: template<> char tmain<char, 1>(char argc, char *argv) {
// CHECK-NEXT: char b = argc, c, d, e, f, g;
// CHECK-NEXT: static char h;
// CHECK-NEXT: S<char> s;
// CHECK-NEXT: char arr[1][10], arr1[1];
// CHECK-NEXT: char i, j, a[20]
// CHECK-NEXT: #pragma omp target parallel
// CHECK-NEXT: h = 2;
// CHECK-NEXT: #pragma omp target parallel default(none) private(argc,b) firstprivate(argv) shared(d) if(parallel: argc > 0) num_threads(1) proc_bind(master) reduction(+: c,arr1[argc]) reduction(max: e,arr[:1][0:10])
// CHECK-NEXT: foo()
// CHECK-NEXT: #pragma omp target parallel if(1) num_threads(s) proc_bind(close) reduction(^: e,f,arr[0:1][:argc]) reduction(&&: g)
// CHECK-NEXT: foo()
// CHECK-NEXT: #pragma omp target parallel if(target: argc > 0)
// CHECK-NEXT: foo()
// CHECK-NEXT: #pragma omp target parallel if(parallel: argc > 0)
// CHECK-NEXT: foo()
// CHECK-NEXT: #pragma omp target parallel if(1)
// CHECK-NEXT: foo()
// CHECK-NEXT: #pragma omp target parallel map(tofrom: i)
// CHECK-NEXT: foo()
// CHECK-NEXT: #pragma omp target parallel map(tofrom: a[0:10],i)
// CHECK-NEXT: foo()
// CHECK-NEXT: #pragma omp target parallel map(to: i) map(from: j)
// CHECK-NEXT: foo()
// CHECK-NEXT: #pragma omp target parallel map(always,alloc: i)
// CHECK-NEXT: foo()
// CHECK-NEXT: #pragma omp target parallel nowait
// CHECK-NEXT: foo()
// CHECK-NEXT: #pragma omp target parallel depend(in : argc,argv[i:argc],a[:])
// CHECK-NEXT: foo()
// CHECK-NEXT: #pragma omp target parallel defaultmap(tofrom: scalar)
// CHECK-NEXT: foo()

// CHECK-LABEL: int main(int argc, char **argv) {
int main (int argc, char **argv) {
  int i, j, a[20];
// CHECK-NEXT: int i, j, a[20]
#pragma omp target parallel
// CHECK-NEXT: #pragma omp target parallel
  foo();
// CHECK-NEXT: foo();
#pragma omp target parallel if (argc > 0)
// CHECK-NEXT: #pragma omp target parallel if(argc > 0)
  foo();
// CHECK-NEXT: foo();

#pragma omp target parallel if (target: argc > 0)
// CHECK-NEXT: #pragma omp target parallel if(target: argc > 0)
  foo();
// CHECK-NEXT: foo();

#pragma omp target parallel if (parallel: argc > 0)
// CHECK-NEXT: #pragma omp target parallel if(parallel: argc > 0)
  foo();
// CHECK-NEXT: foo();

#pragma omp target parallel map(i) if(argc>0)
// CHECK-NEXT: #pragma omp target parallel map(tofrom: i) if(argc > 0)
  foo();
// CHECK-NEXT: foo();

#pragma omp target parallel map(i)
// CHECK-NEXT: #pragma omp target parallel map(tofrom: i)
  foo();
// CHECK-NEXT: foo();

#pragma omp target parallel map(a[0:10], i)
// CHECK-NEXT: #pragma omp target parallel map(tofrom: a[0:10],i)
  foo();
// CHECK-NEXT: foo();

#pragma omp target parallel map(to: i) map(from: j)
// CHECK-NEXT: #pragma omp target parallel map(to: i) map(from: j)
  foo();
// CHECK-NEXT: foo();

#pragma omp target parallel map(always,alloc: i)
// CHECK-NEXT: #pragma omp target parallel map(always,alloc: i)
  foo();
// CHECK-NEXT: foo();

#pragma omp target parallel nowait
// CHECK-NEXT: #pragma omp target parallel nowait
  foo();
// CHECK-NEXT: foo();

#pragma omp target parallel depend(in : argc, argv[i:argc], a[:])
// CHECK-NEXT: #pragma omp target parallel depend(in : argc,argv[i:argc],a[:])
  foo();
// CHECK-NEXT: foo();

#pragma omp target parallel defaultmap(tofrom: scalar)
// CHECK-NEXT: #pragma omp target parallel defaultmap(tofrom: scalar)
  foo();
// CHECK-NEXT: foo();

  return tmain<int, 5>(argc, &argc) + tmain<char, 1>(argv[0][0], argv[0]);
}

extern template int S<int>::TS;
extern template char S<char>::TS;

#endif
