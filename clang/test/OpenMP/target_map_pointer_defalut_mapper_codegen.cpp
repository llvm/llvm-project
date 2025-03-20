// RUN: %clang_cc1 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s

// expected-no-diagnostics
#ifndef HEADER
#define HEADER

typedef struct {
  int a;
} C;
#pragma omp declare mapper(C s) map(to : s.a)

typedef struct {
  int e;
  C f;
  int h;
} D;

D s;

void foo() {
  s.e = 111;
  s.h = 10;
  D *sp = &s;
  D **spp = &sp;
  D ***sppp = &spp;
#pragma omp target map(tofrom : spp[0][0])
  {
    spp[0][0].e = 333;
  }
#pragma omp target map(tofrom : sp[0])
  {
    sp[0].e = 444;
  }
#pragma omp target map(tofrom : sppp[0][0][0])
  {
    sppp[0][0][0].e = 555;
  }
}
#endif

// CHECK: @.offload_sizes = private unnamed_addr constant [5 x i64] [i64 8, i64 0, i64 0, i64 0, i64 4]
// CHECK-NOT: @.offload_sizes = private unnamed_addr constant [6 x i64] [i64 8, i64 0, i64 0, i64 0, i64 8, i64 4]
// CHECK: @.offload_maptypes = private unnamed_addr constant [5 x i64] [i64 35, i64 16, i64 562949953421315, i64 562949953421315, i64 562949953421827]
// CHECK-NOT: .offload_maptypes = private unnamed_addr constant  [6 x i64] [i64 35, i64 0, i64 562949953421315, i64 562949953421315, i64 562949953421827, i64 562949953421843]
// CHECK: @.offload_sizes.1 = private unnamed_addr constant [4 x i64] [i64 0, i64 0, i64 0, i64 4]
// CHECK: @.offload_maptypes.2 = private unnamed_addr constant [4 x i64] [i64 32, i64 281474976710659, i64 281474976710659, i64 281474976711171]
// CHECK: @.offload_sizes.3 = private unnamed_addr constant [6 x i64] [i64 8, i64 8, i64 0, i64 0, i64 0, i64 4]
// CHECK: @.offload_maptypes.4 = private unnamed_addr constant [6 x i64] [i64 35, i64 16, i64 16, i64 844424930131971, i64 844424930131971, i64 844424930132483]
