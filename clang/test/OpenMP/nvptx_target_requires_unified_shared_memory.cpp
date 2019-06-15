// Test declare target link under unified memory requirement.
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-cuda-mode -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

#define N 1000

double var = 10.0;

#pragma omp requires unified_shared_memory
#pragma omp declare target link(var)

int bar(int n){
  double sum = 0;

#pragma omp target
  for(int i = 0; i < n; i++) {
    sum += var;
  }

  return sum;
}

// CHECK: [[VAR:@.+]] = global double 1.000000e+01
// CHECK: [[VAR_DECL_TGT_LINK_PTR:@.+]] = global double* [[VAR]]

// CHECK: [[OFFLOAD_SIZES:@.+]] = private unnamed_addr constant [2 x i64] [i64 4, i64 8]
// CHECK: [[OFFLOAD_MAPTYPES:@.+]] = private unnamed_addr constant [2 x i64] [i64 800, i64 800]

// CHECK: [[N_CASTED:%.+]] = alloca i64
// CHECK: [[SUM_CASTED:%.+]] = alloca i64

// CHECK: [[OFFLOAD_BASEPTRS:%.+]] = alloca [2 x i8*]
// CHECK: [[OFFLOAD_PTRS:%.+]] = alloca [2 x i8*]

// CHECK: [[LOAD1:%.+]] = load i64, i64* [[N_CASTED]]
// CHECK: [[LOAD2:%.+]] = load i64, i64* [[SUM_CASTED]]

// CHECK: [[BPTR1:%.+]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[OFFLOAD_BASEPTRS]], i32 0, i32 0
// CHECK: [[BCAST1:%.+]] = bitcast i8** [[BPTR1]] to i64*
// CHECK: store i64 [[LOAD1]], i64* [[BCAST1]]
// CHECK: [[BPTR2:%.+]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[OFFLOAD_PTRS]], i32 0, i32 0
// CHECK: [[BCAST2:%.+]] = bitcast i8** [[BPTR2]] to i64*
// CHECK: store i64 [[LOAD1]], i64* [[BCAST2]]

// CHECK: [[BPTR3:%.+]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[OFFLOAD_BASEPTRS]], i32 0, i32 1
// CHECK: [[BCAST3:%.+]] = bitcast i8** [[BPTR3]] to i64*
// CHECK: store i64 [[LOAD2]], i64* [[BCAST3]]
// CHECK: [[BPTR4:%.+]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[OFFLOAD_PTRS]], i32 0, i32 1
// CHECK: [[BCAST4:%.+]] = bitcast i8** [[BPTR4]] to i64*
// CHECK: store i64 [[LOAD2]], i64* [[BCAST4]]

// CHECK: [[BPTR7:%.+]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[OFFLOAD_BASEPTRS]], i32 0, i32 0
// CHECK: [[BPTR8:%.+]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[OFFLOAD_PTRS]], i32 0, i32 0

// CHECK: call i32 @__tgt_target(i64 -1, i8* @{{.*}}.region_id, i32 2, i8** [[BPTR7]], i8** [[BPTR8]], i64* getelementptr inbounds ([2 x i64], [2 x i64]* [[OFFLOAD_SIZES]], i32 0, i32 0), i64* getelementptr inbounds ([2 x i64], [2 x i64]* [[OFFLOAD_MAPTYPES]], i32 0, i32 0))

#endif
