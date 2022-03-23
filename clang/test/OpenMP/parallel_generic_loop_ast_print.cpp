// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=51 \
// RUN:   -fsyntax-only -verify %s

// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=51 \
// RUN:   -ast-print %s | FileCheck %s

// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=51 \
// RUN:   -emit-pch -o %t %s

// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=51 \
// RUN:   -include-pch %t -ast-print %s | FileCheck %s

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

//CHECK: template <typename T, int C> void templ_foo(T t) {
//CHECK:   T j, z;
//CHECK:   #pragma omp parallel loop collapse(C) reduction(+: z) lastprivate(j) bind(thread)
//CHECK:   for (T i = 0; i < t; ++i)
//CHECK:       for (j = 0; j < t; ++j)
//CHECK:           z += i + j;
//CHECK: }

//CHECK: template<> void templ_foo<int, 2>(int t) {
//CHECK:     int j, z;
//CHECK:     #pragma omp parallel loop collapse(2) reduction(+: z) lastprivate(j) bind(thread)
//CHECK:         for (int i = 0; i < t; ++i)
//CHECK:             for (j = 0; j < t; ++j)
//CHECK:                 z += i + j;
//CHECK: }
template <typename T, int C>
void templ_foo(T t) {

  T j,z;
  #pragma omp parallel loop collapse(C) reduction(+:z) lastprivate(j) bind(thread)
  for (T i = 0; i<t; ++i)
    for (j = 0; j<t; ++j)
      z += i+j;
}


//CHECK: void test() {
void test() {
  constexpr int N = 100;
  float MTX[N][N];
  int aaa[1000];

  //CHECK: #pragma omp target teams distribute parallel for map(tofrom: MTX)
  //CHECK: #pragma omp parallel loop
  #pragma omp target teams distribute parallel for map(MTX)
  for (auto i = 0; i < N; ++i) {
    #pragma omp parallel loop
    for (auto j = 0; j < N; ++j) {
      MTX[i][j] = 0;
    }
  }

  //CHECK: #pragma omp target teams
  //CHECK: #pragma omp parallel loop
  #pragma omp target teams
  for (int i=0; i<1000; ++i) {
    #pragma omp parallel loop
    for (int j=0; j<100; j++) {
      aaa[i] += i + j;
    }
  }

  int j, z, z1, z2 = 1, z3 = 2;
  //CHECK: #pragma omp parallel loop collapse(2) private(z) lastprivate(j) order(concurrent) firstprivate(z2) num_threads(4) if(1) allocate(omp_default_mem_alloc: z2) shared(z3) reduction(+: z1) bind(parallel) proc_bind(primary)
  #pragma omp parallel loop collapse(2) private(z) lastprivate(j)          \
                   order(concurrent) firstprivate(z2) num_threads(4) if(1) \
                   allocate(omp_default_mem_alloc:z2) shared(z3)           \
                   reduction(+:z1) bind(parallel) proc_bind(primary)
  for (auto i = 0; i < N; ++i) {
    for (j = 0; j < N; ++j) {
      z = i+j;
      MTX[i][j] = z;
      z1 += z;
    }
  }

  //CHECK: #pragma omp target teams
  //CHECK: #pragma omp parallel loop bind(teams)
  #pragma omp target teams
  #pragma omp parallel loop bind(teams)
  for (auto i = 0; i < N; ++i) { }

  //CHECK: #pragma omp target
  //CHECK: #pragma omp teams
  //CHECK: #pragma omp parallel loop bind(teams)
  #pragma omp target
  #pragma omp teams
  #pragma omp parallel loop bind(teams)
  for (auto i = 0; i < N; ++i) { }
}

//CHECK: void nobindingfunc() {
void nobindingfunc()
{
  //CHECK: #pragma omp parallel loop
  #pragma omp parallel loop
  for (int i=0; i<10; ++i) { }
}

void bar()
{
  templ_foo<int,2>(8);
}

#endif // HEADER
