// RUN: %clang_cc1 -verify -fopenmp -triple x86_64-apple-darwin10.6.0 -fopenmp-targets=nvptx64-nvidia-cuda  -emit-llvm-bc -o %t-host.bc %s
// RUN: %clang_cc1 -verify -fopenmp -triple nvptx64-nvidia-cuda -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-host.bc -o - | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

#pragma omp declare target
typedef void **omp_allocator_handle_t;
extern const omp_allocator_handle_t omp_default_mem_alloc;
extern const omp_allocator_handle_t omp_large_cap_mem_alloc;
extern const omp_allocator_handle_t omp_const_mem_alloc;
extern const omp_allocator_handle_t omp_high_bw_mem_alloc;
extern const omp_allocator_handle_t omp_low_lat_mem_alloc;
extern const omp_allocator_handle_t omp_cgroup_mem_alloc;
extern const omp_allocator_handle_t omp_pteam_mem_alloc;
extern const omp_allocator_handle_t omp_thread_mem_alloc;

// CHECK-DAG: @{{.+}}St1{{.+}}b{{.+}} = external global i32,
// CHECK-DAG: @a = global i32 0,
// CHECK-DAG: @b = addrspace(4) global i32 0,
// CHECK-DAG: @c = global i32 0,
// CHECK-DAG: @d = global %struct.St1 zeroinitializer,
// CHECK-DAG: @{{.+}}ns{{.+}}a{{.+}} = addrspace(3) global i32 0,
// CHECK-DAG: @{{.+}}main{{.+}}a{{.*}} = internal global i32 0,
// CHECK-DAG: @{{.+}}ST{{.+}}m{{.+}} = external global i32,
struct St{
 int a;
};

struct St1{
 int a;
 static int b;
#pragma omp allocate(b) allocator(omp_default_mem_alloc)
} d;

int a, b, c;
#pragma omp allocate(a) allocator(omp_large_cap_mem_alloc)
#pragma omp allocate(b) allocator(omp_const_mem_alloc)
#pragma omp allocate(d, c) allocator(omp_high_bw_mem_alloc)

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

namespace ns{
  int a;
}
#pragma omp allocate(ns::a) allocator(omp_pteam_mem_alloc)

// CHECK-LABEL: @main
int main () {
  // CHECK: alloca double,
  static int a;
#pragma omp allocate(a) allocator(omp_thread_mem_alloc)
  a=2;
  double b = 3;
#pragma omp allocate(b)
  return (foo<int>());
}

// CHECK: define {{.*}}i32 @{{.+}}foo{{.+}}()
// CHECK: alloca i32,

extern template int ST<int>::m;
#pragma omp end declare target
#endif
