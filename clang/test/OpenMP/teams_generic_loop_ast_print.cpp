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

//CHECK: template <typename T, int C> void templ_foo(T t) {
//CHECK:   T j, z;
//CHECK:   #pragma omp teams loop collapse(C) reduction(+: z) lastprivate(j) bind(thread) num_teams(C + 2)
//CHECK:   for (T i = 0; i < t; ++i)
//CHECK:       for (j = 0; j < t; ++j)
//CHECK:           z += i + j;
//CHECK: }

//CHECK: template<> void templ_foo<int, 2>(int t) {
//CHECK:     int j, z;
//CHECK:     #pragma omp teams loop collapse(2) reduction(+: z) lastprivate(j) bind(thread) num_teams(2 + 2)
//CHECK:         for (int i = 0; i < t; ++i)
//CHECK:             for (j = 0; j < t; ++j)
//CHECK:                 z += i + j;
//CHECK: }
template <typename T, int C>
void templ_foo(T t) {

  T j,z;
  #pragma omp teams loop collapse(C) reduction(+:z) lastprivate(j) bind(thread) num_teams(C+2)
  for (T i = 0; i<t; ++i)
    for (j = 0; j<t; ++j)
      z += i+j;
}


//CHECK: void test() {
void test() {
  constexpr int N = 100;
  float MTX[N][N];
  int aaa[1000];

  //CHECK: #pragma omp target map(tofrom: MTX)
  //CHECK: #pragma omp teams loop
  #pragma omp target map(MTX)
  #pragma omp teams loop
  for (auto j = 0; j < N; ++j) {
    MTX[0][j] = 0;
  }

  int j, z, z1;
  //CHECK: #pragma omp teams loop collapse(2) private(z) lastprivate(j) order(concurrent) reduction(+: z1) bind(parallel)
  #pragma omp teams loop collapse(2) private(z) lastprivate(j) \
                         order(concurrent) reduction(+:z1) bind(parallel)
  for (auto i = 0; i < N; ++i) {
    for (j = 0; j < N; ++j) {
      z = i+j;
      MTX[i][j] = z;
      z1 += z;
    }
  }

  //CHECK: #pragma omp target
  //CHECK: #pragma omp teams loop bind(teams) num_teams(16) thread_limit(8) default(none)
  #pragma omp target
  #pragma omp teams loop bind(teams) num_teams(16) thread_limit(8) default(none)
  for (auto i = 0; i < N; ++i) { }

  int pr;
  int fpr = 10;
  int k;
  int s = 20;
  //CHECK: #pragma omp target
  //CHECK: #pragma omp teams loop bind(teams) private(pr) firstprivate(fpr) shared(s) allocate(k) reduction(+: k)
  #pragma omp target
  #pragma omp teams loop bind(teams) private(pr) firstprivate(fpr) \
                        shared(s) allocate(k)  reduction(+:k)
  for (auto i = 0; i < N; ++i) {
    pr = i + fpr + s;
  }
}

//CHECK: void nobindingfunc() {
void nobindingfunc()
{
  //CHECK: #pragma omp teams loop
  #pragma omp teams loop
  for (int i=0; i<10; ++i) { }
}

void bar()
{
  templ_foo<int,2>(8);
}

#endif // HEADER
