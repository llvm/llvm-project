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

// &spp[0],    &spp[0],      0,                 IMPLICIT | PARAM
// &spp[0][0], &spp[0][0],   sizeof(spp[0][0]), ALLOC
// &spp[0][0], &spp[0][0].e, sizeof(e),         MEMBER_OF_2 | TO | FROM
// &spp[0][0], &spp[0][0].h, sizeof(h),         MEMBER_OF_2 | TO | FROM
// &spp[0][0], &spp[0][0].f, sizeof(f),         MEMBER_OF_2 | TO | FROM | IMPLICIT, mapper_of_c
// &spp[0],    &spp[0][0],   sizeof(void*),     ATTACH
// CHECK: @.offload_sizes = private unnamed_addr constant [6 x i64] [i64 0, i64 0, i64 0, i64 0, i64 4, i64 8]
// CHECK: @.offload_maptypes = private unnamed_addr constant [6 x i64] [i64 [[#0x220]], i64 0, i64 [[#0x2000000000003]], i64 [[#0x2000000000003]], i64 [[#0x2000000000203]], i64 [[#0x4000]]]

// &sp[0], &sp[0],   sizeof(sp[0]), ALLOC | PARAM
// &sp[0], &sp[0].e, sizeof(e),     MEMBER_OF_1 | TO | FROM
// &sp[0], &sp[0].h, sizeof(h),     MEMBER_OF_1 | TO | FROM
// &sp[0], &sp[0].f, sizeof(f),     MEMBER_OF_1 | TO | FROM | IMPLICIT, mapper_of_c
// &sp,    &sp[0],   sizeof(void*), ATTACH
// CHECK: @.offload_sizes.1 = private unnamed_addr constant [5 x i64] [i64 0, i64 0, i64 0, i64 4, i64 8]
// CHECK: @.offload_maptypes.2 = private unnamed_addr constant [5 x i64] [i64 [[#0x20]], i64 [[#0x1000000000003]], i64 [[#0x1000000000003]], i64 [[#0x1000000000203]], i64 [[#0x4000]]]

// &sppp[0],       &sppp[0],         0,                     IMPLICIT | PARAM
// &sppp[0][0][0], &sppp[0][0][0],   sizeof(sppp[0][0][0]), ALLOC
// &sppp[0][0][0], &sppp[0][0][0].e, sizeof(e),             MEMBER_OF_2 | TO | FROM
// &sppp[0][0][0], &sppp[0][0][0].h, sizeof(h),             MEMBER_OF_2 | TO | FROM
// &sppp[0][0][0], &sppp[0][0][0].f, sizeof(f),             MEMBER_OF_2 | TO | FROM | IMPLICIT, mapper_of_c
// &sppp[0][0],    &sppp[0][0][0],   sizeof(void*),         ATTACH
// CHECK: @.offload_sizes.3 = private unnamed_addr constant [6 x i64] [i64 0, i64 0, i64 0, i64 0, i64 4, i64 8]
// CHECK: @.offload_maptypes.4 = private unnamed_addr constant [6 x i64] [i64 [[#0x220]], i64 0, i64 [[#0x2000000000003]], i64 [[#0x2000000000003]], i64 [[#0x2000000000203]], i64 [[#0x4000]]]
