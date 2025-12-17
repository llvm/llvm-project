// expected-no-diagnostics
#ifndef HEADER
#define HEADER

///==========================================================================///
// RUN: %clang_cc1 -DUSE -DCK21 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefixes=CK21,CK21-64,CK21-USE
// RUN: %clang_cc1 -DUSE -DCK21 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -DUSE -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes=CK21,CK21-64,CK21-USE
// RUN: %clang_cc1 -DUSE -DCK21 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes=CK21,CK21-32,CK21-USE
// RUN: %clang_cc1 -DUSE -DCK21 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -DUSE -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes=CK21,CK21-32,CK21-USE

// RUN: %clang_cc1 -DUSE -DCK21 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefixes=CK21,CK21-64,CK21-USE
// RUN: %clang_cc1 -DUSE -DCK21 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -DUSE -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes=CK21,CK21-64,CK21-USE
// RUN: %clang_cc1 -DUSE -DCK21 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes=CK21,CK21-32,CK21-USE
// RUN: %clang_cc1 -DUSE -DCK21 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -DUSE -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes=CK21,CK21-32,CK21-USE

// RUN: %clang_cc1 -DUSE -DCK21 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefixes=CK21,CK21-64,CK21-USE
// RUN: %clang_cc1 -DUSE -DCK21 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -DUSE -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes=CK21,CK21-64,CK21-USE
// RUN: %clang_cc1 -DUSE -DCK21 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes=CK21,CK21-32,CK21-USE
// RUN: %clang_cc1 -DUSE -DCK21 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -DUSE -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes=CK21,CK21-32,CK21-USE

// RUN: %clang_cc1 -DUSE -DCK21 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY20 %s
// RUN: %clang_cc1 -DUSE -DCK21 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -DUSE -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY20 %s
// RUN: %clang_cc1 -DUSE -DCK21 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY20 %s
// RUN: %clang_cc1 -DUSE -DCK21 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -DUSE -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY20 %s

// RUN: %clang_cc1 -DCK21 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefixes=CK21,CK21-64,CK21-NOUSE
// RUN: %clang_cc1 -DCK21 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes=CK21,CK21-64,CK21-NOUSE
// RUN: %clang_cc1 -DCK21 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes=CK21,CK21-32,CK21-NOUSE
// RUN: %clang_cc1 -DCK21 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes=CK21,CK21-32,CK21-NOUSE

// RUN: %clang_cc1 -DCK21 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefixes=CK21,CK21-64,CK21-NOUSE
// RUN: %clang_cc1 -DCK21 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes=CK21,CK21-64,CK21-NOUSE
// RUN: %clang_cc1 -DCK21 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes=CK21,CK21-32,CK21-NOUSE
// RUN: %clang_cc1 -DCK21 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes=CK21,CK21-32,CK21-NOUSE

// RUN: %clang_cc1 -DCK21 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefixes=CK21,CK21-64,CK21-NOUSE
// RUN: %clang_cc1 -DCK21 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes=CK21,CK21-64,CK21-NOUSE
// RUN: %clang_cc1 -DCK21 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes=CK21,CK21-32,CK21-NOUSE
// RUN: %clang_cc1 -DCK21 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes=CK21,CK21-32,CK21-NOUSE

// RUN: %clang_cc1 -DCK21 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY20 %s
// RUN: %clang_cc1 -DCK21 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY20 %s
// RUN: %clang_cc1 -DCK21 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY20 %s
// RUN: %clang_cc1 -DCK21 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY20 %s

// SIMD-ONLY20-NOT: {{__kmpc|__tgt}}
#ifdef CK21
// CK21: [[ST:%.+]] = type { i32, i32, ptr }

// CK21-LABEL: @.__omp_offloading_{{.*}}foo{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK21: [[SIZE00:@.+]] = private {{.*}}constant [1 x i64] [i64 4]
// CK21-USE: [[MTYPE00:@.+]] = private {{.*}}constant [1 x i64] [i64 35]
// CK21-NOUSE: [[MTYPE00:@.+]] = private {{.*}}constant [1 x i64] [i64 3]

// CK21-LABEL: @.__omp_offloading_{{.*}}foo{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK21: [[SIZE01:@.+]] = private {{.*}}constant [2 x i64] [i64 492, i64 {{4|8}}]
// CK21-USE: [[MTYPE01:@.+]] = private {{.*}}constant [2 x i64] [i64 35, i64 32768]
// CK21-NOUSE: [[MTYPE01:@.+]] = private {{.*}}constant [2 x i64] [i64 3, i64 32768]

// CK21-LABEL: @.__omp_offloading_{{.*}}foo{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK21-USE: [[SIZE02:@.+]] = private {{.*}}constant [3 x i64] [i64 {{4|8}}, i64 500, i64 {{4|8}}]
// CK21-NOUSE: [[SIZE02:@.+]] = private {{.*}}constant [2 x i64] [i64 500, i64 {{4|8}}]
// CK21-USE: [[MTYPE02:@.+]] = private {{.*}}constant [3 x i64] [i64 547, i64 2, i64 32768]
// CK21-NOUSE: [[MTYPE02:@.+]] = private {{.*}}constant [2 x i64] [i64 2, i64 32768]

// CK21-LABEL: @.__omp_offloading_{{.*}}foo{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK21: [[SIZE03:@.+]] = private {{.*}}constant [1 x i64] [i64 492]
// CK21-USE: [[MTYPE03:@.+]] = private {{.*}}constant [1 x i64] [i64 34]
// CK21-NOUSE: [[MTYPE03:@.+]] = private {{.*}}constant [1 x i64] [i64 2]

// CK21-LABEL: @.__omp_offloading_{{.*}}foo{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK21: [[SIZE04:@.+]] = private {{.*}}constant [1 x i64] [i64 4]
// CK21-USE: [[MTYPE04:@.+]] = private {{.*}}constant [1 x i64] [i64 34]
// CK21-NOUSE: [[MTYPE04:@.+]] = private {{.*}}constant [1 x i64] [i64 2]

// CK21-LABEL: @.__omp_offloading_{{.*}}foo{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK21: [[SIZE05:@.+]] = private {{.*}}constant [3 x i64] [i64 0, i64 4, i64 4]
// CK21-USE: [[MTYPE05:@.+]] = private {{.*}}constant [3 x i64] [i64 32, i64 281474976710659, i64 281474976710659]
// CK21-NOUSE: [[MTYPE05:@.+]] = private {{.*}}constant [3 x i64] [i64 0, i64 281474976710659, i64 281474976710659]

// CK21-LABEL: explicit_maps_template_args_and_members{{.*}}(

template <int X, typename T>
struct CC {
  T A;
  int A2;
  float *B;

  int foo(T arg) {
    float la[X];
    T *lb;

// Region 00
// CK21-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK21-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK21-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK21-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK21-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CK21-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK21-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK21-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK21-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK21-DAG: store ptr [[VAR0:%.+]], ptr [[BP0]]
// CK21-DAG: store ptr [[SEC0:%.+]], ptr [[P0]]
// CK21-DAG: [[SEC0]] = getelementptr {{.*}}ptr [[VAR0:%.+]], i{{.+}} 0, i{{.+}} 0

// CK21-USE: call void [[CALL00:@.+]](ptr {{[^,]+}})
// CK21-NOUSE: call void [[CALL00:@.+]]()
#pragma omp target map(A)
    {
#ifdef USE
      A += 1;
#endif
    }

// Region 01

//  &lb[0], &lb[/*lower_bound=*/0], X * sizeof(T), (TO | FROM) / (TO | FROM | PARAM)
//  &lb,    &lb[/*lower_bound=*/0], sizeof(T*),    ATTACH

// CK21-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK21-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK21-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK21-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK21-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CK21-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK21-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK21-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK21-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK21-DAG: store ptr [[RLB:%.+]], ptr [[BP0]]
// CK21-DAG: store ptr [[RLB0:%.+]], ptr [[P0]]
// CK21-DAG: [[RLB]] = load ptr, ptr %lb
// CK21-DAG: [[RLB0]] = getelementptr {{.*}}ptr [[RLB_1:%.+]], i{{.+}} 0
// CK21-DAG: [[RLB_1]] = load ptr, ptr %lb

// CK21-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
// CK21-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
// CK21-DAG: store ptr %lb, ptr [[BP1]]
// CK21-DAG: store ptr [[RLB0_1:%.+]], ptr [[P1]]
// CK21-DAG: [[RLB0_1]] = getelementptr {{.*}}ptr [[RLB_2:%.+]], i{{.+}} 0
// CK21-DAG: [[RLB_2]] = load ptr, ptr %lb

// CK21-USE: call void [[CALL01:@.+]](ptr {{[^,]+}})
// CK21-NOUSE: call void [[CALL01:@.+]]()
#pragma omp target map(lb[:X])
    {
#ifdef USE
      lb[4] += 1;
#endif
    }

// Region 02

//  [&this[0], &this[0].B, sizeof(B),     PARAM | ALLOC | IMPLICIT]
//  &B[0],     &B[X],      2 * sizeof(T), FROM
//  &B,        &B[X],      sizeof(T*),    ATTACH

// CK21-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK21-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK21-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK21-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK21-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CK21-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK21-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK21-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK21-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK21-USE-DAG: store ptr [[THIS:%.+]], ptr [[BP0]]
// CK21-USE-DAG: store ptr [[B:%.+]], ptr [[P0]]
// CK21-USE-DAG: [[B:%B]] = getelementptr inbounds {{.*}}[[THIS]], i{{.*}} 0, i{{.*}} 2

// CK21-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
// CK21-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
// CK21-USE-DAG: store ptr [[VAR0:%.+]], ptr [[BP1]]
// CK21-USE-DAG: store ptr [[SEC0:%.+]], ptr [[P1]]
// CK21-NOUSE-DAG: store ptr [[VAR0:%.+]], ptr [[BP0]]
// CK21-NOUSE-DAG: store ptr [[SEC0:%.+]], ptr [[P0]]

// CK21-DAG: [[VAR0]] = load ptr, ptr [[VAR00:%[^,]+]]
// CK21-DAG: [[VAR00]] = getelementptr inbounds nuw %struct.CC, ptr [[THIS:%.+]], i{{.*}} 0, i{{.*}} 2
// CK21-DAG: [[SEC0]] = getelementptr inbounds nuw float, ptr [[SEC00:%.+]], i{{.*}} 123
// CK21-DAG: [[SEC00]] = load ptr, ptr [[SEC000:%[^,]+]]
// CK21-DAG: [[SEC000]] = getelementptr inbounds nuw %struct.CC, ptr [[THIS]], i{{.*}} 0, i{{.*}} 2

// CK21-USE-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 2
// CK21-USE-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 2
// CK21-USE-DAG: store ptr [[VAR00:%.+]], ptr [[BP2]]
// CK21-USE-DAG: store ptr [[SEC0:%.+]], ptr [[P2]]
// CK21-NOUSE-DAG: store ptr [[VAR00]], ptr [[BP1]]
// CK21-NOUSE-DAG: store ptr [[SEC0]], ptr [[P1]]

// CK21-USE: call void [[CALL02:@.+]](ptr {{[^,]+}})
// CK21-NOUSE: call void [[CALL02:@.+]]()
#pragma omp target map(from \
                       : B [X:X + 2])
    {
#ifdef USE
      B[2] += 1.0f;
#endif
    }

// Region 03
// CK21-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK21-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK21-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK21-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK21-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CK21-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK21-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK21-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK21-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK21-DAG: store ptr [[VAR0:%.+]], ptr [[BP0]]
// CK21-DAG: store ptr [[VAR0]], ptr [[P0]]

// CK21-USE: call void [[CALL03:@.+]](ptr {{[^,]+}})
// CK21-NOUSE: call void [[CALL03:@.+]]()
#pragma omp target map(from \
                       : la)
    {
#ifdef USE
      la[3] += 1.0f;
#endif
    }

// Region 04
// CK21-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK21-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK21-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK21-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK21-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CK21-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK21-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK21-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK21-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK21-DAG: store ptr [[VAR0:%.+]], ptr [[BP0]]
// CK21-DAG: store ptr [[VAR0]], ptr [[P0]]

// CK21-USE: call void [[CALL04:@.+]](ptr {{[^,]+}})
// CK21-NOUSE: call void [[CALL04:@.+]]()
#pragma omp target map(from \
                       : arg)
    {
#ifdef USE
      arg +=1;
#endif
    }

// Make sure the extra flag is passed to the second map.
// Region 05
// CK21-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK21-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK21-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK21-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK21-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CK21-DAG: [[SARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 4
// CK21-DAG: store ptr [[SIZES:%.+]], ptr [[SARG]]
// CK21-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK21-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
// CK21-DAG: [[SIZES]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

// CK21-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK21-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK21-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
// CK21-DAG: store ptr [[VAR0:%.+]], ptr [[BP0]]
// CK21-DAG: store ptr [[SEC0:%.+]], ptr [[P0]]
// CK21-DAG: store i64 {{%.+}}, ptr [[S0]]
// CK21-DAG: [[SEC0]] = getelementptr {{.*}}ptr [[VAR0]], i{{.+}} 0, i{{.+}} 0

// CK21-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
// CK21-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
// CK21-DAG: store ptr [[VAR0]], ptr [[BP1]]
// CK21-DAG: store ptr [[SEC0]], ptr [[P1]]

// CK21-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 2
// CK21-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 2
// CK21-DAG: store ptr [[VAR2:%.+]], ptr [[BP2]]
// CK21-DAG: store ptr [[SEC2:%.+]], ptr [[P2]]
// CK21-DAG: [[SEC2]] = getelementptr {{.*}}ptr [[VAR2]], i{{.+}} 0, i{{.+}} 1

// CK21-USE: call void [[CALL05:@.+]](ptr {{[^,]+}})
// CK21-NOUSE: call void [[CALL05:@.+]]()
#pragma omp target map(A, A2)
    {
#ifdef USE
      A += 1;
      A2 += 1;
#endif
    }
    return A;
  }
};

int explicit_maps_template_args_and_members(int a){
  CC<123,int> c;
  return c.foo(a);
}

// CK21: define {{.+}}[[CALL00]]
// CK21: define {{.+}}[[CALL01]]
// CK21: define {{.+}}[[CALL02]]
// CK21: define {{.+}}[[CALL03]]
// CK21: define {{.+}}[[CALL04]]
// CK21: define {{.+}}[[CALL05]]
#endif // CK21
#endif
