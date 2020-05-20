// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=50 -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=50 -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

struct omp_alloctrait_t {};

typedef void **omp_allocator_handle_t;
extern const omp_allocator_handle_t omp_null_allocator;
extern const omp_allocator_handle_t omp_default_mem_alloc;
extern const omp_allocator_handle_t omp_large_cap_mem_alloc;
extern const omp_allocator_handle_t omp_const_mem_alloc;
extern const omp_allocator_handle_t omp_high_bw_mem_alloc;
extern const omp_allocator_handle_t omp_low_lat_mem_alloc;
extern const omp_allocator_handle_t omp_cgroup_mem_alloc;
extern const omp_allocator_handle_t omp_pteam_mem_alloc;
extern const omp_allocator_handle_t omp_thread_mem_alloc;

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
  omp_alloctrait_t traits[10];
  omp_allocator_handle_t my_allocator;
#pragma omp target teams
  a=2;
#pragma omp target teams default(none), private(argc,b) firstprivate(argv) shared (d) reduction(+:c) reduction(max:e) num_teams(C) thread_limit(d*C) allocate(argv)
  foo();
#pragma omp target teams allocate(my_allocator:f) reduction(^:e, f) reduction(&& : g) uses_allocators(my_allocator(traits))
  foo();
  return 0;
}

// CHECK: template <typename T, int C> T tmain(T argc, T *argv) {
// CHECK-NEXT: T b = argc, c, d, e, f, g;
// CHECK-NEXT: static T a;
// CHECK-NEXT: S<T> s;
// CHECK-NEXT: omp_alloctrait_t traits[10];
// CHECK-NEXT: omp_allocator_handle_t my_allocator;
// CHECK-NEXT: #pragma omp target teams{{$}}
// CHECK-NEXT: a = 2;
// CHECK-NEXT: #pragma omp target teams default(none) private(argc,b) firstprivate(argv) shared(d) reduction(+: c) reduction(max: e) num_teams(C) thread_limit(d * C) allocate(argv)
// CHECK-NEXT: foo()
// CHECK-NEXT: #pragma omp target teams allocate(my_allocator: f) reduction(^: e,f) reduction(&&: g) uses_allocators(my_allocator(traits))
// CHECK-NEXT: foo()
// CHECK: template<> int tmain<int, 5>(int argc, int *argv) {
// CHECK-NEXT: int b = argc, c, d, e, f, g;
// CHECK-NEXT: static int a;
// CHECK-NEXT: S<int> s;
// CHECK-NEXT: omp_alloctrait_t traits[10];
// CHECK-NEXT: omp_allocator_handle_t my_allocator;
// CHECK-NEXT: #pragma omp target teams
// CHECK-NEXT: a = 2;
// CHECK-NEXT: #pragma omp target teams default(none) private(argc,b) firstprivate(argv) shared(d) reduction(+: c) reduction(max: e) num_teams(5) thread_limit(d * 5) allocate(argv)
// CHECK-NEXT: foo()
// CHECK-NEXT: #pragma omp target teams allocate(my_allocator: f) reduction(^: e,f) reduction(&&: g) uses_allocators(my_allocator(traits))
// CHECK-NEXT: foo()
// CHECK: template<> long tmain<long, 1>(long argc, long *argv) {
// CHECK-NEXT: long b = argc, c, d, e, f, g;
// CHECK-NEXT: static long a;
// CHECK-NEXT: S<long> s;
// CHECK-NEXT: omp_alloctrait_t traits[10];
// CHECK-NEXT: omp_allocator_handle_t my_allocator;
// CHECK-NEXT: #pragma omp target teams
// CHECK-NEXT: a = 2;
// CHECK-NEXT: #pragma omp target teams default(none) private(argc,b) firstprivate(argv) shared(d) reduction(+: c) reduction(max: e) num_teams(1) thread_limit(d * 1) allocate(argv)
// CHECK-NEXT: foo()
// CHECK-NEXT: #pragma omp target teams allocate(my_allocator: f) reduction(^: e,f) reduction(&&: g) uses_allocators(my_allocator(traits))
// CHECK-NEXT: foo()

enum Enum { };

int main (int argc, char **argv) {
  long x;
  int b = argc, c, d, e, f, g;
  static int a;
  #pragma omp threadprivate(a)
  Enum ee;
// CHECK: Enum ee;
#pragma omp target teams
// CHECK-NEXT: #pragma omp target teams
  a=2;
// CHECK-NEXT: a = 2;
#pragma omp target teams default(none), private(argc,b) num_teams(f) firstprivate(argv) reduction(| : c, d) reduction(* : e) thread_limit(f+g)
// CHECK-NEXT: #pragma omp target teams default(none) private(argc,b) num_teams(f) firstprivate(argv) reduction(|: c,d) reduction(*: e) thread_limit(f + g)
  foo();
// CHECK-NEXT: foo();
  return tmain<int, 5>(b, &b) + tmain<long, 1>(x, &x);
}

extern template int S<int>::TS;
extern template long S<long>::TS;
#endif
