// expected-no-diagnostics
#ifndef HEADER
#define HEADER

///==========================================================================///
// RUN: %clang_cc1 -DCK30 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK30 --check-prefix CK30-64
// RUN: %clang_cc1 -DCK30 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK30 --check-prefix CK30-64
// RUN: %clang_cc1 -DCK30 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK30 --check-prefix CK30-32
// RUN: %clang_cc1 -DCK30 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK30 --check-prefix CK30-32

// RUN: %clang_cc1 -DCK30 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK30 --check-prefix CK30-64
// RUN: %clang_cc1 -DCK30 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK30 --check-prefix CK30-64
// RUN: %clang_cc1 -DCK30 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK30 --check-prefix CK30-32
// RUN: %clang_cc1 -DCK30 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK30 --check-prefix CK30-32

// RUN: %clang_cc1 -DCK30 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK30 --check-prefix CK30-64
// RUN: %clang_cc1 -DCK30 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK30 --check-prefix CK30-64
// RUN: %clang_cc1 -DCK30 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK30 --check-prefix CK30-32
// RUN: %clang_cc1 -DCK30 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK30 --check-prefix CK30-32

// RUN: %clang_cc1 -DCK30 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY30 %s
// RUN: %clang_cc1 -DCK30 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY30 %s
// RUN: %clang_cc1 -DCK30 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY30 %s
// RUN: %clang_cc1 -DCK30 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY30 %s
// SIMD-ONLY30-NOT: {{__kmpc|__tgt}}
#ifdef CK30

// CK30-DAG: [[BASE:%.+]] = type { ptr, i32, ptr }
// CK30-DAG: [[STRUCT:%.+]] = type { [[BASE]], ptr, ptr, i32, ptr }

// &s,             &s,             sizeof(s),             TO | FROM | PARAM
// &s.ptr1[0],     &s.ptr1[0],     sizeof(s.ptr1[0]),     TO | FROM
// &s.ptr1,        &s.ptr1[0],     sizeof(void*),         ATTACH
// &s.ptrBase1[0], &s.ptrBase1[0], sizeof(s.ptrBase1[0]), TO | FROM
// &s.ptrBase1,    &s.ptrBase1[0], sizeof(void*),         ATTACH

// CK30-LABEL: @.__omp_offloading_{{.*}}map_with_deep_copy{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK30: [[SIZE00:@.+]] = private unnamed_addr constant [5 x i64] [i64 {{56|28}}, i64 4, i64 {{4|8}}, i64 4, i64 {{4|8}}]
// CK30: [[MTYPE00:@.+]] = private unnamed_addr constant [5 x i64] [i64 [[#0x23]], i64 3, i64 [[#0x4000]], i64 3, i64 [[#0x4000]]]

typedef struct {
  int *ptrBase;
  int valBase;
  int *ptrBase1;
} Base;

typedef struct StructWithPtrTag : public Base {
  int *ptr;
  int *ptr2;
  int val;
  int *ptr1;
} StructWithPtr;

// CK30-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK30-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK30-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK30-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK30-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CK30-DAG: [[PGEP]] = getelementptr inbounds [5 x ptr], ptr [[PTRS:%.+]], i32 0, i32 0
// CK30-DAG: [[BPGEP]] = getelementptr inbounds [5 x ptr], ptr [[BASES:%.+]], i32 0, i32 0

// CK30-DAG: [[BASE_PTR:%.+]] = getelementptr inbounds [5 x ptr], ptr [[BASES]], i32 0, i32 0
// CK30-DAG: store ptr [[S:%.+]], ptr [[BASE_PTR]],
// CK30-DAG: [[PTR:%.+]] = getelementptr inbounds [5 x ptr], ptr [[PTRS]], i32 0, i32 0
// CK30-DAG: store ptr [[S]], ptr [[PTR]],

// CK30-DAG: [[BASE_PTR:%.+]] = getelementptr inbounds [5 x ptr], ptr [[BASES]], i32 0, i32 1
// CK30-DAG: [[PTR:%.+]] = getelementptr inbounds [5 x ptr], ptr [[PTRS]], i32 0, i32 1
// CK30-DAG: store ptr [[S_PTR1_REF:%.+]], ptr [[BASE_PTR]],
// CK30-DAG: store ptr [[S_PTR1_BEGIN:%.+]], ptr [[PTR]],
// CK30-DAG: [[S_PTR1_REF]] = load ptr, ptr [[S_PTR1:%.+]],
// CK30-DAG: [[S_PTR1]] = getelementptr inbounds nuw [[STRUCT]], ptr [[S]], i32 0, i32 4
// CK30-DAG: [[S_PTR1_BEGIN]] = getelementptr inbounds nuw i32, ptr [[S_PTR1_REF1:%.+]], i{{64|32}} 0
// CK30-DAG: [[S_PTR1_REF1]] = load ptr, ptr [[S_PTR11:%.+]],
// CK30-DAG: [[S_PTR11]] = getelementptr inbounds nuw [[STRUCT]], ptr [[S]], i32 0, i32 4

// CK30-DAG: [[BASE_PTR:%.+]] = getelementptr inbounds [5 x ptr], ptr [[BASES]], i32 0, i32 2
// CK30-DAG: [[PTR:%.+]] = getelementptr inbounds [5 x ptr], ptr [[PTRS]], i32 0, i32 2
// CK30-DAG: store ptr [[S_PTR1]], ptr [[BASE_PTR]],
// CK30-DAG: store ptr [[S_PTR1_BEGIN]], ptr [[PTR]],

// CK30-DAG: [[BASE_PTR:%.+]] = getelementptr inbounds [5 x ptr], ptr [[BASES]], i32 0, i32 3
// CK30-DAG: [[PTR:%.+]] = getelementptr inbounds [5 x ptr], ptr [[PTRS]], i32 0, i32 3
// CK30-DAG: store ptr [[S_PTRBASE1_REF:%.+]], ptr [[BASE_PTR]],
// CK30-DAG: store ptr [[S_PTRBASE1_BEGIN:%.+]], ptr [[PTR]],
// CK30-DAG: [[S_PTRBASE1_REF]] = load ptr, ptr [[S_PTRBASE1:%.+]],
// CK30-DAG: [[S_PTRBASE1]] = getelementptr inbounds nuw [[BASE]], ptr [[S_BASE:%.+]], i32 0, i32 2
// CK30-DAG: [[S_PTRBASE1_BEGIN]] = getelementptr inbounds nuw i32, ptr [[S_PTRBASE1_REF1:%.+]], i{{64|32}} 0
// CK30-DAG: [[S_PTRBASE1_REF1]] = load ptr, ptr [[S_PTRBASE11:%.+]],
// CK30-DAG: [[S_PTRBASE11]] = getelementptr inbounds nuw [[BASE]], ptr [[S_BASE:%.+]], i32 0, i32 2

// CK30-DAG: [[BASE_PTR:%.+]] = getelementptr inbounds [5 x ptr], ptr [[BASES]], i32 0, i32 4
// CK30-DAG: [[PTR:%.+]] = getelementptr inbounds [5 x ptr], ptr [[PTRS]], i32 0, i32 4
// CK30-DAG: store ptr [[S_PTRBASE1]], ptr [[BASE_PTR]],
// CK30-DAG: store ptr [[S_PTRBASE1_BEGIN]], ptr [[PTR]],

void map_with_deep_copy() {
  StructWithPtr s;
#pragma omp target map(s, s.ptr1 [0:1], s.ptrBase1 [0:1])
  {
    s.val++;
    s.ptr1[0]++;
    s.ptrBase1[0] = 10001;
  }
}

#endif // CK30
#endif
