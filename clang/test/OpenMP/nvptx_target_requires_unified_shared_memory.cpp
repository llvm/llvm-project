// Test declare target link under unified memory requirement.

// RUN: %clang_cc1 -verify -fopenmp -fopenmp-cuda-mode -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK-HOST

// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -fopenmp-cuda-mode -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -fopenmp-cuda-mode -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -target-cpu sm_70 -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s --check-prefix CHECK-DEVICE

// Test declare target link under unified memory requirement.

// RUN: %clang_cc1 -verify -fopenmp -fopenmp-cuda-mode -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK-HOST

// RUN: %clang_cc1 -verify -fopenmp -fopenmp-cuda-mode -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-cuda-mode -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -target-cpu sm_70 -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s --check-prefix CHECK-DEVICE

// expected-no-diagnostics

#ifndef HEADER
#define HEADER

#define N 1000

double var = 10.0;
double to_var = 20.0;

#pragma omp requires unified_shared_memory
#pragma omp declare target link(var)
#pragma omp declare target to(to_var)

int bar(int n){
  double sum = 0;

#pragma omp target
  for(int i = 0; i < n; i++) {
    sum += var + to_var;
  }

  return sum;
}

// CHECK-HOST: [[VAR:@.+]] ={{.*}} global double 1.000000e+01
// CHECK-HOST: [[VAR_DECL_TGT_LINK_PTR:@.+]] = weak{{.*}} global ptr [[VAR]]

// CHECK-HOST: [[TO_VAR:@.+]] ={{.*}} global double 2.000000e+01
// CHECK-HOST: [[VAR_DECL_TGT_TO_PTR:@.+]] = weak{{.*}} global ptr [[TO_VAR]]

// CHECK-HOST: [[OFFLOAD_SIZES:@.+]] = private unnamed_addr constant [2 x i64] [i64 4, i64 8]
// CHECK-HOST: [[OFFLOAD_MAPTYPES:@.+]] = private unnamed_addr constant [2 x i64] [i64 800, i64 800]

// CHECK-HOST: [[OMP_OFFLOAD_ENTRY_LINK_VAR_PTR_NAME:@.+]] = internal unnamed_addr constant [21 x i8]
// CHECK-HOST: [[OMP_OFFLOAD_ENTRY_LINK_VAR_PTR:@.+]] = weak constant %struct.__tgt_offload_entry { i64 0, i16 1, i16 1, i32 1, ptr [[VAR_DECL_TGT_LINK_PTR]], ptr [[OMP_OFFLOAD_ENTRY_LINK_VAR_PTR_NAME]], i64 8, i64 0, ptr null }, section "omp_offl

// CHECK-HOST: [[OMP_OFFLOAD_ENTRY_TO_VAR_PTR_NAME:@.+]] = internal unnamed_addr constant [24 x i8]
// CHECK-HOST: [[OMP_OFFLOAD_ENTRY_TO_VAR_PTR:@.+]] = weak constant %struct.__tgt_offload_entry { i64 0, i16 1, i16 1, i32 0, ptr [[VAR_DECL_TGT_TO_PTR]], ptr [[OMP_OFFLOAD_ENTRY_TO_VAR_PTR_NAME]], i64 8, i64 0, ptr null }, section "omp_offloading_entries", align 1

// CHECK-HOST: [[N_CASTED:%.+]] = alloca i64
// CHECK-HOST: [[SUM_CASTED:%.+]] = alloca i64

// CHECK-HOST: [[OFFLOAD_BASEPTRS:%.+]] = alloca [2 x ptr]
// CHECK-HOST: [[OFFLOAD_PTRS:%.+]] = alloca [2 x ptr]

// CHECK-HOST: [[LOAD1:%.+]] = load i64, ptr [[N_CASTED]]
// CHECK-HOST: [[LOAD2:%.+]] = load i64, ptr [[SUM_CASTED]]

// CHECK-HOST: [[BPTR1:%.+]] = getelementptr inbounds [2 x ptr], ptr [[OFFLOAD_BASEPTRS]], i32 0, i32 0
// CHECK-HOST: store i64 [[LOAD1]], ptr [[BPTR1]]
// CHECK-HOST: [[BPTR2:%.+]] = getelementptr inbounds [2 x ptr], ptr [[OFFLOAD_PTRS]], i32 0, i32 0
// CHECK-HOST: store i64 [[LOAD1]], ptr [[BPTR2]]

// CHECK-HOST: [[BPTR3:%.+]] = getelementptr inbounds [2 x ptr], ptr [[OFFLOAD_BASEPTRS]], i32 0, i32 1
// CHECK-HOST: store i64 [[LOAD2]], ptr [[BPTR3]]
// CHECK-HOST: [[BPTR4:%.+]] = getelementptr inbounds [2 x ptr], ptr [[OFFLOAD_PTRS]], i32 0, i32 1
// CHECK-HOST: store i64 [[LOAD2]], ptr [[BPTR4]]

// CHECK-HOST: [[BPTR7:%.+]] = getelementptr inbounds [2 x ptr], ptr [[OFFLOAD_BASEPTRS]], i32 0, i32 0
// CHECK-HOST: [[BPTR8:%.+]] = getelementptr inbounds [2 x ptr], ptr [[OFFLOAD_PTRS]], i32 0, i32 0

// CHECK-HOST: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr %{{.+}})

// CHECK-DEVICE: [[VAR_LINK:@.+]] = weak{{.*}} global ptr null
// CHECK-DEVICE: [[VAR_TO:@.+]] = weak{{.*}} global ptr null

// CHECK-DEVICE: [[VAR_TO_PTR:%.+]] = load ptr, ptr [[VAR_TO]]
// CHECK-DEVICE: load double, ptr [[VAR_TO_PTR]]

#endif
