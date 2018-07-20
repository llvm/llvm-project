// RUN: %clang_cc1 -verify -fopenmp -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp-simd -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s
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
// CHECK:      template<> struct S<long> {
// CHECK:        static long TS;
// CHECK-NEXT:   #pragma omp threadprivate(S<long>::TS)
// CHECK-NEXT: }

template <typename T, int C>
T tmain(T argc, T *argv) {
  T b = argc, c, d, e, f, g;
  static T a;
  S<T> s;
#pragma omp target
#pragma omp teams
  a=2;
#pragma omp target
#pragma omp teams default(none), private(argc,b) firstprivate(argv) shared (d) reduction(+:c) reduction(max:e) num_teams(C) thread_limit(d*C)
  foo();
#pragma omp target
#pragma omp teams reduction(^:e, f) reduction(&& : g)
  foo();
  return 0;
}

// CHECK: template <typename T, int C> T tmain(T argc, T *argv) {
// CHECK-NEXT: T b = argc, c, d, e, f, g;
// CHECK-NEXT: static T a;
// CHECK-NEXT: S<T> s;
// CHECK-NEXT: #pragma omp target
// CHECK-NEXT: #pragma omp teams{{$}}
// CHECK-NEXT: a = 2;
// CHECK-NEXT: #pragma omp target
// CHECK-NEXT: #pragma omp teams default(none) private(argc,b) firstprivate(argv) shared(d) reduction(+: c) reduction(max: e) num_teams(C) thread_limit(d * C)
// CHECK-NEXT: foo()
// CHECK-NEXT: #pragma omp target
// CHECK-NEXT: #pragma omp teams reduction(^: e,f) reduction(&&: g)
// CHECK-NEXT: foo()
// CHECK: template<> int tmain<int, 5>(int argc, int *argv) {
// CHECK-NEXT: int b = argc, c, d, e, f, g;
// CHECK-NEXT: static int a;
// CHECK-NEXT: S<int> s;
// CHECK-NEXT: #pragma omp target
// CHECK-NEXT: #pragma omp teams
// CHECK-NEXT: a = 2;
// CHECK-NEXT: #pragma omp target
// CHECK-NEXT: #pragma omp teams default(none) private(argc,b) firstprivate(argv) shared(d) reduction(+: c) reduction(max: e) num_teams(5) thread_limit(d * 5)
// CHECK-NEXT: foo()
// CHECK-NEXT: #pragma omp target
// CHECK-NEXT: #pragma omp teams reduction(^: e,f) reduction(&&: g)
// CHECK-NEXT: foo()
// CHECK: template<> long tmain<long, 1>(long argc, long *argv) {
// CHECK-NEXT: long b = argc, c, d, e, f, g;
// CHECK-NEXT: static long a;
// CHECK-NEXT: S<long> s;
// CHECK-NEXT: #pragma omp target
// CHECK-NEXT: #pragma omp teams
// CHECK-NEXT: a = 2;
// CHECK-NEXT: #pragma omp target
// CHECK-NEXT: #pragma omp teams default(none) private(argc,b) firstprivate(argv) shared(d) reduction(+: c) reduction(max: e) num_teams(1) thread_limit(d * 1)
// CHECK-NEXT: foo()
// CHECK-NEXT: #pragma omp target
// CHECK-NEXT: #pragma omp teams reduction(^: e,f) reduction(&&: g)
// CHECK-NEXT: foo()

enum Enum { };

int main (int argc, char **argv) {
  long x;
  int b = argc, c, d, e, f, g;
  static int a;
  #pragma omp threadprivate(a)
  Enum ee;
// CHECK: Enum ee;
#pragma omp target
#pragma omp teams
// CHECK-NEXT: #pragma omp target
// CHECK-NEXT: #pragma omp teams
  a=2;
// CHECK-NEXT: a = 2;
#pragma omp target
#pragma omp teams default(none), private(argc,b) num_teams(f) firstprivate(argv) reduction(| : c, d) reduction(* : e) thread_limit(f+g)
// CHECK-NEXT: #pragma omp target
// CHECK-NEXT: #pragma omp teams default(none) private(argc,b) num_teams(f) firstprivate(argv) reduction(|: c,d) reduction(*: e) thread_limit(f + g)
  foo();
// CHECK-NEXT: foo();
  return tmain<int, 5>(b, &b) + tmain<long, 1>(x, &x);
}

extern template int S<int>::TS;
extern template long S<long>::TS;
#endif
