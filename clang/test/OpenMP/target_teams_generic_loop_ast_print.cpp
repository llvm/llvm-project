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

//CHECK: template <typename T, int C, int D> void templ_foo(T t) {
//CHECK:   T j, z;
//CHECK:   #pragma omp target teams loop device(D) collapse(C) reduction(+: z) lastprivate(j) bind(thread) num_teams(C + 2)
//CHECK:   for (T i = 0; i < t; ++i)
//CHECK:       for (j = 0; j < t; ++j)
//CHECK:           z += i + j;
//CHECK: }

//CHECK: template<> void templ_foo<int, 2, 0>(int t) {
//CHECK:     int j, z;
//CHECK:     #pragma omp target teams loop device(0) collapse(2) reduction(+: z) lastprivate(j) bind(thread) num_teams(2 + 2)
//CHECK:         for (int i = 0; i < t; ++i)
//CHECK:             for (j = 0; j < t; ++j)
//CHECK:                 z += i + j;
//CHECK: }
template <typename T, int C, int D>
void templ_foo(T t) {

  T j,z;
  #pragma omp target teams loop device(D) collapse(C) reduction(+:z) lastprivate(j) bind(thread) num_teams(C+2)
  for (T i = 0; i<t; ++i)
    for (j = 0; j<t; ++j)
      z += i+j;
}


//CHECK: void test() {
void test() {
  constexpr int N = 100;
  float MTX[N][N];
  int aaa[1000];

  //CHECK: #pragma omp target teams loop map(tofrom: MTX)
  #pragma omp target teams loop map(MTX)
  for (auto j = 0; j < N; ++j) {
    MTX[0][j] = 0;
  }

  int j, z, z1;
  //CHECK: #pragma omp target teams loop collapse(2) private(z) lastprivate(j) order(concurrent) reduction(+: z1) bind(parallel)
  #pragma omp target teams loop collapse(2) private(z) lastprivate(j) \
                         order(concurrent) reduction(+:z1) bind(parallel)
  for (auto i = 0; i < N; ++i) {
    for (j = 0; j < N; ++j) {
      z = i+j;
      MTX[i][j] = z;
      z1 += z;
    }
  }

  //CHECK: #pragma omp target teams loop bind(teams) num_teams(16) thread_limit(8) default(none)
  #pragma omp target teams loop bind(teams) num_teams(16) thread_limit(8) default(none)
  for (auto i = 0; i < N; ++i) { }

  int pr;
  int zzz;
  //CHECK: #pragma omp target teams loop private(zzz) uses_allocators(omp_default_mem_alloc) allocate(omp_default_mem_alloc: zzz) if(1) device(0) map(tofrom: pr)
  #pragma omp target teams loop private(zzz) uses_allocators(omp_default_mem_alloc) allocate(omp_default_mem_alloc:zzz) if(1) device(0) map(tofrom:pr)
  for (int i=0; i<1000; ++i) {
    zzz = i + 1;
    pr = 33;
  }

  int fpr = 10;
  int k;
  int s = 20;
  //CHECK: #pragma omp target teams loop bind(teams) private(pr) firstprivate(fpr) shared(s) allocate(k) reduction(+: k)
  #pragma omp target teams loop bind(teams) private(pr) firstprivate(fpr) \
                        shared(s) allocate(k)  reduction(+:k)
  for (auto i = 0; i < N; ++i) {
    pr = i + fpr + s;
  }

  short y = 3;
  //CHECK: #pragma omp target teams loop map(tofrom: y) depend(out : y)
  #pragma omp target teams loop map(tofrom:y) depend(out:y)
  for (int i=0; i<10; ++i) {
    y = 3+i;
  }
}

//CHECK: void nobindingfunc() {
void nobindingfunc()
{
  //CHECK: #pragma omp target teams loop
  #pragma omp target teams loop
  for (int i=0; i<10; ++i) { }
}

void bar()
{
  templ_foo<int,2,0>(8);
}

#endif // HEADER
