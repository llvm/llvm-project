// RUN: %clang_cc1 -verify -fopenmp -triple x86_64-apple-darwin10.6.0 -ast-print %s -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -triple x86_64-apple-darwin10.6.0 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -triple x86_64-apple-darwin10.6.0 -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp -triple x86_64-unknown-linux-gnu -ast-print %s -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -fnoopenmp-use-tls -triple x86_64-unknown-linux-gnu -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fnoopenmp-use-tls -triple x86_64-unknown-linux-gnu -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print -o - | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp-simd -triple x86_64-apple-darwin10.6.0 -ast-print %s -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp-simd -triple x86_64-apple-darwin10.6.0 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -triple x86_64-apple-darwin10.6.0 -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp-simd -triple x86_64-unknown-linux-gnu -ast-print %s -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp-simd -fnoopenmp-use-tls -triple x86_64-unknown-linux-gnu -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fnoopenmp-use-tls -triple x86_64-unknown-linux-gnu -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print -o - | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

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

struct St{
 int a;
};

struct St1{
 int a;
 static int b;
// CHECK: static int b;
#pragma omp allocate(b) allocator(omp_default_mem_alloc)
// CHECK-NEXT: #pragma omp allocate(St1::b) allocator(omp_default_mem_alloc){{$}}
} d;

int a, b, c;
// CHECK: int a;
// CHECK: int b;
// CHECK: int c;
#pragma omp allocate(a) allocator(omp_large_cap_mem_alloc)
#pragma omp allocate(b) allocator(omp_const_mem_alloc)
// CHECK-NEXT: #pragma omp allocate(a) allocator(omp_large_cap_mem_alloc)
// CHECK-NEXT: #pragma omp allocate(b) allocator(omp_const_mem_alloc)
#pragma omp allocate(c, d) allocator(omp_high_bw_mem_alloc)
// CHECK-NEXT: #pragma omp allocate(c,d) allocator(omp_high_bw_mem_alloc)

template <class T>
struct ST {
  static T m;
  #pragma omp allocate(m) allocator(omp_low_lat_mem_alloc)
};

template <class T> T foo() {
  T v;
  #pragma omp allocate(v) allocator(omp_cgroup_mem_alloc)
  v = ST<T>::m;
  return v;
}
//CHECK: template <class T> T foo() {
//CHECK-NEXT: T v;
//CHECK-NEXT: #pragma omp allocate(v) allocator(omp_cgroup_mem_alloc)
//CHECK: template<> int foo<int>() {
//CHECK-NEXT: int v;
//CHECK-NEXT: #pragma omp allocate(v) allocator(omp_cgroup_mem_alloc)

namespace ns{
  int a;
}
// CHECK: namespace ns {
// CHECK-NEXT: int a;
// CHECK-NEXT: }
#pragma omp allocate(ns::a) allocator(omp_pteam_mem_alloc)
// CHECK-NEXT: #pragma omp allocate(ns::a) allocator(omp_pteam_mem_alloc)

int main () {
  static int a;
// CHECK: static int a;
#pragma omp allocate(a) allocator(omp_thread_mem_alloc)
// CHECK-NEXT: #pragma omp allocate(a) allocator(omp_thread_mem_alloc)
  a=2;
  int b = 3;
// CHECK: int b = 3;
#pragma omp allocate(b)
// CHECK-NEXT: #pragma omp allocate(b)
  return (foo<int>());
}

extern template int ST<int>::m;
#endif
