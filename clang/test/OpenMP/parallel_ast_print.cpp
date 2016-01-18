// RUN: %clang_cc1 -verify -fopenmp -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s
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

// CHECK:      template <class T = int> struct S {
// CHECK:        static int TS;
// CHECK-NEXT:   #pragma omp threadprivate(S<int>::TS)
// CHECK-NEXT: }
// CHECK:      template <class T = long> struct S {
// CHECK:        static long TS;
// CHECK-NEXT:   #pragma omp threadprivate(S<long>::TS)
// CHECK-NEXT: }
// CHECK:      template <class T> struct S {
// CHECK:        static T TS;
// CHECK-NEXT:   #pragma omp threadprivate(S::TS)
// CHECK:      };

template <typename T, int C>
T tmain(T argc, T *argv) {
  T b = argc, c, d, e, f, g;
  static T a;
  S<T> s;
  T arr[C][10], arr1[C];
#pragma omp parallel
  a=2;
#pragma omp parallel default(none), private(argc,b) firstprivate(argv) shared (d) if (parallel:argc > 0) num_threads(C) copyin(S<T>::TS) proc_bind(master) reduction(+:c, arr1[argc]) reduction(max:e, arr[:C][0:10])
  foo();
#pragma omp parallel if (C) num_threads(s) proc_bind(close) reduction(^:e, f, arr[0:C][:argc]) reduction(&& : g)
  foo();
  return 0;
}

// CHECK: template <typename T = int, int C = 5> int tmain(int argc, int *argv) {
// CHECK-NEXT: int b = argc, c, d, e, f, g;
// CHECK-NEXT: static int a;
// CHECK-NEXT: S<int> s;
// CHECK-NEXT: int arr[5][10], arr1[5];
// CHECK-NEXT: #pragma omp parallel
// CHECK-NEXT: a = 2;
// CHECK-NEXT: #pragma omp parallel default(none) private(argc,b) firstprivate(argv) shared(d) if(parallel: argc > 0) num_threads(5) copyin(S<int>::TS) proc_bind(master) reduction(+: c,arr1[argc]) reduction(max: e,arr[:5][0:10])
// CHECK-NEXT: foo()
// CHECK-NEXT: #pragma omp parallel if(5) num_threads(s) proc_bind(close) reduction(^: e,f,arr[0:5][:argc]) reduction(&&: g)
// CHECK-NEXT: foo()
// CHECK: template <typename T = long, int C = 1> long tmain(long argc, long *argv) {
// CHECK-NEXT: long b = argc, c, d, e, f, g;
// CHECK-NEXT: static long a;
// CHECK-NEXT: S<long> s;
// CHECK-NEXT: long arr[1][10], arr1[1];
// CHECK-NEXT: #pragma omp parallel
// CHECK-NEXT: a = 2;
// CHECK-NEXT: #pragma omp parallel default(none) private(argc,b) firstprivate(argv) shared(d) if(parallel: argc > 0) num_threads(1) copyin(S<long>::TS) proc_bind(master) reduction(+: c,arr1[argc]) reduction(max: e,arr[:1][0:10])
// CHECK-NEXT: foo()
// CHECK-NEXT: #pragma omp parallel if(1) num_threads(s) proc_bind(close) reduction(^: e,f,arr[0:1][:argc]) reduction(&&: g)
// CHECK-NEXT: foo()
// CHECK: template <typename T, int C> T tmain(T argc, T *argv) {
// CHECK-NEXT: T b = argc, c, d, e, f, g;
// CHECK-NEXT: static T a;
// CHECK-NEXT: S<T> s;
// CHECK-NEXT: T arr[C][10], arr1[C];
// CHECK-NEXT: #pragma omp parallel
// CHECK-NEXT: a = 2;
// CHECK-NEXT: #pragma omp parallel default(none) private(argc,b) firstprivate(argv) shared(d) if(parallel: argc > 0) num_threads(C) copyin(S<T>::TS) proc_bind(master) reduction(+: c,arr1[argc]) reduction(max: e,arr[:C][0:10])
// CHECK-NEXT: foo()
// CHECK-NEXT: #pragma omp parallel if(C) num_threads(s) proc_bind(close) reduction(^: e,f,arr[0:C][:argc]) reduction(&&: g)
// CHECK-NEXT: foo()

enum Enum { };

int main (int argc, char **argv) {
  long x;
  int b = argc, c, d, e, f, g;
  static int a;
  #pragma omp threadprivate(a)
  int arr[10][argc], arr1[2];
  Enum ee;
// CHECK: Enum ee;
#pragma omp parallel
// CHECK-NEXT: #pragma omp parallel
  a=2;
// CHECK-NEXT: a = 2;
#pragma omp parallel default(none), private(argc,b) firstprivate(argv) if (parallel: argc > 0) num_threads(ee) copyin(a) proc_bind(spread) reduction(| : c, d, arr1[argc]) reduction(* : e, arr[:10][0:argc])
// CHECK-NEXT: #pragma omp parallel default(none) private(argc,b) firstprivate(argv) if(parallel: argc > 0) num_threads(ee) copyin(a) proc_bind(spread) reduction(|: c,d,arr1[argc]) reduction(*: e,arr[:10][0:argc])
  foo();
// CHECK-NEXT: foo();
// CHECK-NEXT: #pragma omp parallel if(b) num_threads(c) proc_bind(close) reduction(^: e,f) reduction(&&: g,arr[0:argc][:10])
// CHECK-NEXT: foo()
#pragma omp parallel if (b) num_threads(c) proc_bind(close) reduction(^:e, f) reduction(&& : g, arr[0:argc][:10])
  foo();
  return tmain<int, 5>(b, &b) + tmain<long, 1>(x, &x);
}

#endif
