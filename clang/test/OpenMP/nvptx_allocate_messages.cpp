// RUN: %clang_cc1 -verify -fopenmp -triple x86_64-apple-darwin10.6.0 -fopenmp-targets=nvptx64-nvidia-cuda  -emit-llvm-bc -o %t-host.bc %s
// RUN: %clang_cc1 -verify -DDEVICE -fopenmp -triple nvptx64-nvidia-cuda -fopenmp-targets=nvptx64-nvidia-cuda -fsyntax-only %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-host.bc
// RUN: %clang_cc1 -verify -DDEVICE -DREQUIRES -fopenmp -triple nvptx64-nvidia-cuda -fopenmp-targets=nvptx64-nvidia-cuda -fsyntax-only %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-host.bc
#if !defined(DEVICE) || defined(REQUIRES)
// expected-no-diagnostics
#endif // DEVICE

#ifndef HEADER
#define HEADER

#if defined(REQUIRES) && defined(DEVICE)
#pragma omp requires dynamic_allocators
#endif // REQUIRES && DEVICE

int bar() {
  int res = 0;
#if defined(DEVICE) && !defined(REQUIRES)
// expected-error@+2 {{expected an 'allocator' clause inside of the target region; provide an 'allocator' clause or use 'requires' directive with the 'dynamic_allocators' clause}}
#endif // DEVICE && !REQUIRES
#pragma omp allocate(res)
  return 0;
}

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
#if defined(DEVICE) && !defined(REQUIRES)
// expected-error@+2 2 {{expected an allocator expression inside of the target region; provide an allocator expression or use 'requires' directive with the 'dynamic_allocators' clause}}
#endif // DEVICE && !REQUIRES
#pragma omp parallel private(v) allocate(v)
  v = 0;
  return v;
}

namespace ns{
  int a;
}
#pragma omp allocate(ns::a) allocator(omp_pteam_mem_alloc)

int main () {
  static int a;
#pragma omp allocate(a) allocator(omp_thread_mem_alloc)
  a=2;
  double b = 3;
#if defined(DEVICE) && !defined(REQUIRES)
// expected-error@+2 {{expected an 'allocator' clause inside of the target region; provide an 'allocator' clause or use 'requires' directive with the 'dynamic_allocators' clause}}
#endif // DEVICE && !REQUIRES
#pragma omp allocate(b)
#if defined(DEVICE) && !defined(REQUIRES)
// expected-note@+3 {{in instantiation of function template specialization 'foo<int>' requested here}}
// expected-note@+2 {{called by 'main'}}
#endif // DEVICE && !REQUIRES
  return (foo<int>() + bar());
}

extern template int ST<int>::m;
#pragma omp end declare target
#endif
