// RUN: %clang_cc1 -verify -fopenmp -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

void foo() {}

template <class T>
struct S {
  operator T() { return T(); }
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
#pragma omp parallel sections
  {
    a = 2;
  }
#pragma omp parallel sections default(none), private(argc, b) firstprivate(argv) shared(d) if (parallel: argc > 0) num_threads(C) copyin(S < T > ::TS) proc_bind(master) reduction(+ : c) reduction(max : e)
  {
    foo();
  }
#pragma omp parallel sections if (C) num_threads(s) proc_bind(close) reduction(^ : e, f) reduction(&& : g) lastprivate(b, c)
  {
    foo();
#pragma omp section
    foo();
  }
  return 0;
}

// CHECK: template <typename T = int, int C = 5> int tmain(int argc, int *argv) {
// CHECK-NEXT: int b = argc, c, d, e, f, g;
// CHECK-NEXT: static int a;
// CHECK-NEXT: S<int> s;
// CHECK-NEXT: #pragma omp parallel sections
// CHECK-NEXT: {
// CHECK-NEXT: a = 2;
// CHECK-NEXT: }
// CHECK-NEXT: #pragma omp parallel sections default(none) private(argc,b) firstprivate(argv) shared(d) if(parallel: argc > 0) num_threads(5) copyin(S<int>::TS) proc_bind(master) reduction(+: c) reduction(max: e)
// CHECK-NEXT: {
// CHECK-NEXT: foo();
// CHECK-NEXT: }
// CHECK-NEXT: #pragma omp parallel sections if(5) num_threads(s) proc_bind(close) reduction(^: e,f) reduction(&&: g) lastprivate(b,c)
// CHECK-NEXT: {
// CHECK-NEXT: foo();
// CHECK-NEXT: #pragma omp section
// CHECK-NEXT: foo();
// CHECK-NEXT: }
// CHECK: template <typename T = long, int C = 1> long tmain(long argc, long *argv) {
// CHECK-NEXT: long b = argc, c, d, e, f, g;
// CHECK-NEXT: static long a;
// CHECK-NEXT: S<long> s;
// CHECK-NEXT: #pragma omp parallel sections
// CHECK-NEXT: {
// CHECK-NEXT: a = 2;
// CHECK-NEXT: }
// CHECK-NEXT: #pragma omp parallel sections default(none) private(argc,b) firstprivate(argv) shared(d) if(parallel: argc > 0) num_threads(1) copyin(S<long>::TS) proc_bind(master) reduction(+: c) reduction(max: e)
// CHECK-NEXT: {
// CHECK-NEXT: foo();
// CHECK-NEXT: }
// CHECK-NEXT: #pragma omp parallel sections if(1) num_threads(s) proc_bind(close) reduction(^: e,f) reduction(&&: g) lastprivate(b,c)
// CHECK-NEXT: {
// CHECK-NEXT: foo();
// CHECK-NEXT: #pragma omp section
// CHECK-NEXT: foo();
// CHECK-NEXT: }
// CHECK: template <typename T, int C> T tmain(T argc, T *argv) {
// CHECK-NEXT: T b = argc, c, d, e, f, g;
// CHECK-NEXT: static T a;
// CHECK-NEXT: S<T> s;
// CHECK-NEXT: #pragma omp parallel sections
// CHECK-NEXT: {
// CHECK-NEXT: a = 2;
// CHECK-NEXT: }
// CHECK-NEXT: #pragma omp parallel sections default(none) private(argc,b) firstprivate(argv) shared(d) if(parallel: argc > 0) num_threads(C) copyin(S<T>::TS) proc_bind(master) reduction(+: c) reduction(max: e)
// CHECK-NEXT: {
// CHECK-NEXT: foo();
// CHECK-NEXT: }
// CHECK-NEXT: #pragma omp parallel sections if(C) num_threads(s) proc_bind(close) reduction(^: e,f) reduction(&&: g) lastprivate(b,c)
// CHECK-NEXT: {
// CHECK-NEXT: foo();
// CHECK-NEXT: #pragma omp section
// CHECK-NEXT: foo();
// CHECK-NEXT: }

enum Enum {};

int main(int argc, char **argv) {
  long x;
  int b = argc, c, d, e, f, g;
  static int a;
#pragma omp threadprivate(a)
  Enum ee;
// CHECK: Enum ee;
#pragma omp parallel sections
  // CHECK-NEXT: #pragma omp parallel sections
  {
    a = 2;
  }
// CHECK-NEXT: {
// CHECK-NEXT: a = 2;
// CHECK-NEXT: }
#pragma omp parallel sections default(none), private(argc, b) firstprivate(argv) if (argc > 0) num_threads(ee) copyin(a) proc_bind(spread) reduction(| : c, d) reduction(* : e) lastprivate(argv)
  // CHECK-NEXT: #pragma omp parallel sections default(none) private(argc,b) firstprivate(argv) if(argc > 0) num_threads(ee) copyin(a) proc_bind(spread) reduction(|: c,d) reduction(*: e) lastprivate(argv)
  {
    foo();
#pragma omp section
    foo();
#pragma omp section
    foo();
  }
  // CHECK-NEXT: {
  // CHECK-NEXT: foo();
  // CHECK-NEXT: #pragma omp section
  // CHECK-NEXT: foo();
  // CHECK-NEXT: #pragma omp section
  // CHECK-NEXT: foo();
  // CHECK-NEXT: }
  return tmain<int, 5>(b, &b) + tmain<long, 1>(x, &x);
}

template<typename T>
T S<T>::TS = 0;

#endif
