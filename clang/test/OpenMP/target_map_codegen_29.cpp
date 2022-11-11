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

// CK30-LABEL: @.__omp_offloading_{{.*}}map_with_deep_copy{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK30: [[SIZE00:@.+]] = private unnamed_addr constant [4 x i64] [i64 0, i64 {{56|28}}, i64 4, i64 4]
// The first element: 0x20 - OMP_MAP_TARGET_PARAM
// 2: 0x1000000000003 - OMP_MAP_MEMBER_OF(0) | OMP_MAP_TO | OMP_MAP_FROM - copies all the data in structs excluding deep-copied elements (from &s to end of s).
// 3-4: 0x1000000000013 - OMP_MAP_MEMBER_OF(0) | OMP_MAP_PTR_AND_OBJ | OMP_MAP_TO | OMP_MAP_FROM - deep copy of the pointers + pointee.
// CK30: [[MTYPE00:@.+]] = private {{.*}}constant [4 x i64] [i64 32, i64 281474976710659, i64 281474976710675, i64 281474976710675]

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
// CK30-DAG: [[SARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 4
// CK30-DAG: store ptr [[SIZES:%.+]], ptr [[SARG]]
// CK30-DAG: [[SIZES]] = getelementptr inbounds [4 x i{{64|32}}], ptr [[SIZES:%.+]], i32 0, i32 0
// CK30-DAG: [[PGEP]] = getelementptr inbounds [4 x ptr], ptr [[PTRS:%.+]], i32 0, i32 0
// CK30-DAG: [[BPGEP]] = getelementptr inbounds [4 x ptr], ptr [[BASES:%.+]], i32 0, i32 0

// CK30-DAG: [[BASE_PTR:%.+]] = getelementptr inbounds [4 x ptr], ptr [[BASES]], i32 0, i32 0
// CK30-DAG: store ptr [[S:%.+]], ptr [[BASE_PTR]],
// CK30-DAG: [[PTR:%.+]] = getelementptr inbounds [4 x ptr], ptr [[PTRS]], i32 0, i32 0
// CK30-DAG: store ptr [[S]], ptr [[PTR]],
// CK30-DAG: [[SIZE:%.+]] = getelementptr inbounds [4 x i{{64|32}}], ptr [[SIZES]], i32 0, i32 0
// CK30-DAG: store i64 [[S_ALLOC_SIZE:%.+]], ptr [[SIZE]],
// CK30-DAG: [[S_ALLOC_SIZE]] = sdiv exact i64 [[DIFF:%.+]], ptrtoint (ptr getelementptr (i8, ptr null, i32 1) to i64)
// CK30-DAG: [[DIFF]] = sub i64 [[S_END_BC:%.+]], [[S_BEGIN_BC:%.+]]
// CK30-DAG: [[S_BEGIN_BC]] = ptrtoint ptr [[S_BEGIN:%.+]] to i64
// CK30-DAG: [[S_END_BC]] = ptrtoint ptr [[S_END:%.+]] to i64
// CK30-DAG: [[REAL_S_END:%.+]] = getelementptr [[STRUCT]], ptr [[S]], i32 1

// CK30-DAG: [[BASE_PTR:%.+]] = getelementptr inbounds [4 x ptr], ptr [[BASES]], i32 0, i32 1
// CK30-DAG: store ptr [[S]], ptr [[BASE_PTR]],
// CK30-DAG: [[PTR:%.+]] = getelementptr inbounds [4 x ptr], ptr [[PTRS]], i32 0, i32 1
// CK30-DAG: store ptr [[S]], ptr [[PTR]],

// CK30-DAG: [[BASE_PTR:%.+]] = getelementptr inbounds [4 x ptr], ptr [[BASES]], i32 0, i32 2
// CK30-DAG: store ptr [[S_PTR1:%.+]], ptr [[BASE_PTR]],
// CK30-DAG: [[PTR:%.+]] = getelementptr inbounds [4 x ptr], ptr [[PTRS]], i32 0, i32 2
// CK30-DAG: store ptr [[S_PTR1_BEGIN:%.+]], ptr [[PTR]],
// CK30-DAG: [[S_PTR1]] = getelementptr inbounds [[STRUCT]], ptr [[S]], i32 0, i32 4
// CK30-DAG: [[S_PTR1_BEGIN]] = getelementptr inbounds i32, ptr [[S_PTR1_BEGIN_REF:%.+]], i{{64|32}} 0
// CK30-DAG: [[S_PTR1_BEGIN_REF]] = load ptr, ptr [[S_PTR1:%.+]],
// CK30-DAG: [[S_PTR1]] = getelementptr inbounds [[STRUCT]], ptr [[S]], i32 0, i32 4

// CK30-DAG: [[BASE_PTR:%.+]] = getelementptr inbounds [4 x ptr], ptr [[BASES]], i32 0, i32 3
// CK30-DAG: store ptr [[S_PTRBASE1:%.+]], ptr [[BASE_PTR]],
// CK30-DAG: [[PTR:%.+]] = getelementptr inbounds [4 x ptr], ptr [[PTRS]], i32 0, i32 3
// CK30-DAG: store ptr [[S_PTRBASE1_BEGIN:%.+]], ptr [[PTR]],
// CK30-DAG: [[S_PTRBASE1]] = getelementptr inbounds [[BASE]], ptr [[S_BASE:%.+]], i32 0, i32 2
// CK30-DAG: [[S_PTRBASE1_BEGIN]] = getelementptr inbounds i32, ptr [[S_PTRBASE1_BEGIN_REF:%.+]], i{{64|32}} 0
// CK30-DAG: [[S_PTRBASE1_BEGIN_REF]] = load ptr, ptr [[S_PTRBASE1:%.+]],
// CK30-DAG: [[S_PTRBASE1]] = getelementptr inbounds [[BASE]], ptr [[S_BASE:%.+]], i32 0, i32 2
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
