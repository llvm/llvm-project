// expected-no-diagnostics
#ifndef HEADER
#define HEADER

///==========================================================================///
// RUN: %clang_cc1 -DCK24 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK24 --check-prefix CK24-64
// RUN: %clang_cc1 -DCK24 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK24 --check-prefix CK24-64
// RUN: %clang_cc1 -DCK24 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK24 --check-prefix CK24-32
// RUN: %clang_cc1 -DCK24 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK24 --check-prefix CK24-32

// RUN: %clang_cc1 -DCK24 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK24 --check-prefix CK24-64
// RUN: %clang_cc1 -DCK24 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK24 --check-prefix CK24-64
// RUN: %clang_cc1 -DCK24 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK24 --check-prefix CK24-32
// RUN: %clang_cc1 -DCK24 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK24 --check-prefix CK24-32

// RUN: %clang_cc1 -DCK24 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK24 --check-prefix CK24-64
// RUN: %clang_cc1 -DCK24 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK24 --check-prefix CK24-64
// RUN: %clang_cc1 -DCK24 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK24 --check-prefix CK24-32
// RUN: %clang_cc1 -DCK24 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK24 --check-prefix CK24-32

// RUN: %clang_cc1 -DCK24 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY23 %s
// RUN: %clang_cc1 -DCK24 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY23 %s
// RUN: %clang_cc1 -DCK24 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY23 %s
// RUN: %clang_cc1 -DCK24 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY23 %s
// SIMD-ONLY23-NOT: {{__kmpc|__tgt}}
#ifdef CK24

// CK24-DAG: [[SC:%.+]] = type { i32, [[SB:%.+]], ptr, [10 x i32] }
// CK24-DAG: [[SB]] = type { i32, [[SA:%.+]], [10 x [[SA:%.+]]], [10 x ptr], ptr }
// CK24-DAG: [[SA]] = type { i32, ptr, [10 x i32] }

struct SA{
  int a;
  struct SA *p;
  int b[10];
};
struct SB{
  int a;
  struct SA s;
  struct SA sa[10];
  struct SA *sp[10];
  struct SA *p;
};
struct SC{
  int a;
  struct SB s;
  struct SB *p;
  int b[10];
};

// CK24-LABEL: @.__omp_offloading_{{.*}}explicit_maps_struct_fields{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK24: [[SIZE01:@.+]] = private {{.*}}constant [1 x i64] [i64 4]
// CK24: [[MTYPE01:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK24-LABEL: @.__omp_offloading_{{.*}}explicit_maps_struct_fields{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK24: [[SIZE13:@.+]] = private {{.*}}constant [2 x i64] [i64 4, i64 {{4|8}}]
// CK24: [[MTYPE13:@.+]] = private {{.*}}constant [2 x i64] [i64 35, i64 32768]

// CK24-LABEL: @.__omp_offloading_{{.*}}explicit_maps_struct_fields{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK24: [[SIZE14:@.+]] = private {{.*}}constant [2 x i64] [i64 {{48|56}}, i64 {{4|8}}]
// CK24: [[MTYPE14:@.+]] = private {{.*}}constant [2 x i64] [i64 35, i64 32768]

// CK24-LABEL: @.__omp_offloading_{{.*}}explicit_maps_struct_fields{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK24: [[SIZE15:@.+]] = private {{.*}}constant [2 x i64] [i64 4, i64 {{4|8}}]
// CK24: [[MTYPE15:@.+]] = private {{.*}}constant [2 x i64] [i64 35, i64 32768]

// CK24-LABEL: @.__omp_offloading_{{.*}}explicit_maps_struct_fields{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK24: [[SIZE16:@.+]] = private {{.*}}constant [3 x i64] [i64 0, i64 20, i64 {{4|8}}]
// CK24: [[MTYPE16:@.+]] = private {{.*}}constant [3 x i64] [i64 32, i64 281474976710659, i64 32768]

// CK24-LABEL: @.__omp_offloading_{{.*}}explicit_maps_struct_fields{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK24: [[SIZE17:@.+]] = private {{.*}}constant [3 x i64] [i64 0, i64 {{3560|2880}}, i64 {{4|8}}]
// CK24: [[MTYPE17:@.+]] = private {{.*}}constant [3 x i64] [i64 544, i64 3, i64 32768]

// CK24-LABEL: @.__omp_offloading_{{.*}}explicit_maps_struct_fields{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK24: [[SIZE18:@.+]] = private {{.*}}constant [2 x i64] [i64 4, i64 {{4|8}}]
// CK24: [[MTYPE18:@.+]] = private {{.*}}constant [2 x i64] [i64 35, i64 32768]

// CK24-LABEL: @.__omp_offloading_{{.*}}explicit_maps_struct_fields{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK24: [[SIZE19:@.+]] = private unnamed_addr constant [3 x i64] [i64 0, i64 4, i64 {{4|8}}]
// CK24: [[MTYPE19:@.+]] = private unnamed_addr constant [3 x i64] [i64 544, i64 3, i64 32768]

// CK24-LABEL: @.__omp_offloading_{{.*}}explicit_maps_struct_fields{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK24: [[SIZE20:@.+]] = private unnamed_addr constant [3 x i64] [i64 0, i64 4, i64 {{4|8}}]
// CK24: [[MTYPE20:@.+]] = private unnamed_addr constant [3 x i64] [i64 544, i64 3, i64 32768]

// CK24-LABEL: @.__omp_offloading_{{.*}}explicit_maps_struct_fields{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK24: [[SIZE21:@.+]] = private unnamed_addr constant [3 x i64] [i64 0, i64 4, i64 {{4|8}}]
// CK24: [[MTYPE21:@.+]] = private unnamed_addr constant [3 x i64] [i64 544, i64 3, i64 32768]

// CK24-LABEL: @.__omp_offloading_{{.*}}explicit_maps_struct_fields{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK24: [[SIZE22:@.+]] = private {{.*}}constant [3 x i64] [i64 0, i64 8, i64 {{4|8}}]
// CK24: [[MTYPE22:@.+]] = private {{.*}}constant [3 x i64] [i64 32, i64 281474976710659, i64 32768]

// CK24-LABEL: @.__omp_offloading_{{.*}}explicit_maps_struct_fields{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK24: [[SIZE23:@.+]] = private unnamed_addr constant [4 x i64] [i64 0, i64 0, i64 8, i64 {{4|8}}]
// CK24: [[MTYPE23:@.+]] = private unnamed_addr constant [4 x i64] [i64 544, i64 0, i64 562949953421315, i64 32768]

// CK24-LABEL: @.__omp_offloading_{{.*}}explicit_maps_struct_fields{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK24: [[SIZE24:@.+]] = private unnamed_addr constant [3 x i64] [i64 0, i64 4, i64 {{4|8}}]
// CK24: [[MTYPE24:@.+]] = private unnamed_addr constant [3 x i64] [i64 544, i64 3, i64 32768]

// CK24-LABEL: explicit_maps_struct_fields
int explicit_maps_struct_fields(int a){
  SC s;
  SC *p;

// Region 01
// CK24-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK24-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK24-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK24-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK24-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CK24-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK24-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK24-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: store ptr [[VAR0:%.+]], ptr [[BP0]]
// CK24-DAG: store ptr [[SEC0:%.+]], ptr [[P0]]
// CK24-DAG: [[SEC0]] = getelementptr {{.*}}ptr [[VAR0]], i{{.+}} 0, i{{.+}} 0

// CK24: call void [[CALL01:@.+]](ptr {{[^,]+}})
#pragma omp target map(s.a)
  { s.a++; }

//
// Same thing but starting from a pointer.

// Region 13

//  &p[0], &p->a, sizeof(p->a), TO | FROM | PARAM
//  &p,    &p->a, sizeof(p),    ATTACH

// CK24-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK24-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK24-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK24-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK24-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CK24-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK24-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK24-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: store ptr [[VAR0:%.+]], ptr [[BP0]]
// CK24-DAG: store ptr [[SEC0:%.+]], ptr [[P0]]
// CK24-DAG: [[SEC0]] = getelementptr {{.*}}ptr [[VAR00:%.+]], i{{.+}} 0, i{{.+}} 0

// CK24-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: store ptr [[VAR:[^,]+]], ptr [[BP1]]
// CK24-DAG: store ptr [[SEC1:%.+]], ptr [[P1]]
// CK24-DAG: [[SEC1]] = getelementptr {{.*}}ptr [[VAR1:%.+]], i{{.+}} 0, i{{.+}} 0

// CK24-DAG: [[VAR0]] = load ptr, ptr [[VAR]]
// CK24-DAG: [[VAR00]] = load ptr, ptr [[VAR]]
// CK24-DAG: [[VAR1]] = load ptr, ptr [[VAR]]


// CK24: call void [[CALL13:@.+]](ptr {{[^,]+}})
#pragma omp target map(p->a)
  { p->a++; }

// Region 14

//  &p[0], &p->s.s, sizeof(p->s.s), TO | FROM | PARAM
//  &p,    &p->s.s, sizeof(p),      ATTACH

// CK24-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK24-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK24-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK24-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK24-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CK24-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK24-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK24-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: store ptr [[VAR0:%.+]], ptr [[BP0]]
// CK24-DAG: store ptr [[SEC0:%.+]], ptr [[P0]]
// CK24-DAG: [[SEC0]] = getelementptr {{.*}}ptr [[SEC00:%[^,]+]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[SEC00]] = getelementptr {{.*}}ptr [[VAR00:%.+]], i{{.+}} 0, i{{.+}} 1

// CK24-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: store ptr [[VAR:%[^,]+]], ptr [[BP1]]
// CK24-DAG: store ptr [[SEC1:%.+]], ptr [[P1]]
// CK24-DAG: [[SEC1]] = getelementptr {{.*}}ptr [[SEC11:%[^,]+]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[SEC11]] = getelementptr {{.*}}ptr [[VAR1:%.+]], i{{.+}} 0, i{{.+}} 1

// CK24-DAG: [[VAR0]] = load ptr, ptr [[VAR]]
// CK24-DAG: [[VAR00]] = load ptr, ptr [[VAR]]
// CK24-DAG: [[VAR1]] = load ptr, ptr [[VAR]]
// CK24: call void [[CALL14:@.+]](ptr {{[^,]+}})
#pragma omp target map(p->s.s)
  { p->a++; }

// Region 15

//  &p[0], &p->s.s.a, sizeof(p->s.s.a), TO | FROM | PARAM
//  &p, &p->s.s.a, sizeof(p), ATTACH

// CK24-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK24-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK24-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK24-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK24-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CK24-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK24-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK24-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: store ptr [[VAR0:%.+]], ptr [[BP0]]
// CK24-DAG: store ptr [[SEC0:%.+]], ptr [[P0]]
// CK24-DAG: [[SEC0]] = getelementptr {{.*}}ptr [[SEC00:%[^,]+]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[SEC00]] = getelementptr {{.*}}ptr [[SEC000:%[^,]+]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[SEC000]] = getelementptr {{.*}}ptr [[VAR00:%.+]], i{{.+}} 0, i{{.+}} 1

// CK24-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: store ptr [[VAR:%[^,]+]], ptr [[BP1]]
// CK24-DAG: store ptr [[SEC1:%.+]], ptr [[P1]]
// CK24-DAG: [[SEC1]] = getelementptr {{.*}}ptr [[SEC11:%[^,]+]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[SEC11]] = getelementptr {{.*}}ptr [[SEC111:%[^,]+]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[SEC111]] = getelementptr {{.*}}ptr [[VAR1:%.+]], i{{.+}} 0, i{{.+}} 1

// CK24-DAG: [[VAR0]] = load ptr, ptr [[VAR]]
// CK24-DAG: [[VAR00]] = load ptr, ptr [[VAR]]
// CK24-DAG: [[VAR1]] = load ptr, ptr [[VAR]]

// CK24: call void [[CALL15:@.+]](ptr {{[^,]+}})
#pragma omp target map(p->s.s.a)
  { p->a++; }

// Region 16

//  &p[0], &p->b[0], sizeof(p->b[0:5]), ALLOC | PARAM
//  &p[0], &p->b[0], sizeof(p->b[0:5]), TO | FROM
//  &p,    &p->b[0], sizeof(p),         ATTACH

// CK24-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK24-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK24-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK24-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK24-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CK24-DAG: [[SARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 4
// CK24-DAG: store ptr [[SIZES:%.+]], ptr [[SARG]]
// CK24-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK24-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
// CK24-DAG: [[SIZES]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

// CK24-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: store ptr [[VAR0:%.+]], ptr [[BP0]]
// CK24-DAG: store ptr [[SEC0:%.+]], ptr [[P0]]
// CK24-DAG: store i64 {{%.+}}, ptr [[S0]]
// CK24-DAG: [[SEC0]] = getelementptr {{.*}}ptr [[SEC00:%[^,]+]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[SEC00]] = getelementptr {{.*}}ptr [[VAR00:%.+]], i{{.+}} 0, i{{.+}} 3

// CK24-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: store ptr [[VAR0]], ptr [[BP1]]
// CK24-DAG: store ptr [[SEC0]], ptr [[P1]]

// CK24-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 2
// CK24-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 2
// CK24-DAG: store ptr [[VAR:%[^,]+]], ptr [[BP2]]
// CK24-DAG: store ptr [[SEC0]], ptr [[P2]]

// CK24-DAG: [[VAR0]] = load ptr, ptr [[VAR]]
// CK24-DAG: [[VAR00]] = load ptr, ptr [[VAR]]

// CK24: call void [[CALL16:@.+]](ptr {{[^,]+}})
#pragma omp target map(p->b[:5])
  { p->a++; }

// Region 17

//  &p[0],    &p[0],           0,                 IMPLICIT | PARAM
//  &p->p[0], &p->p[/*lb=*/0], sizeof(p->p[0:5]), TO | FROM
//  &p->p,    &p->p[/*lb=*/0], sizeof(p),         ATTACH

// CK24-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK24-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK24-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK24-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK24-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CK24-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK24-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK24-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: store ptr [[VAR0:%.+]], ptr [[BP0]]
// CK24-DAG: store ptr [[VAR0]], ptr [[P0]]
// CK24-DAG: [[VAR0]] = load ptr, ptr %p

// CK24-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: store ptr [[VAR1:%.+]], ptr [[BP1]]
// CK24-DAG: store ptr [[SEC1:%.+]], ptr [[P1]]
// CK24-DAG: [[VAR1]] = load ptr, ptr [[VAR11:%[^,]+]],
// CK24-DAG: [[VAR11]] = getelementptr {{.*}}%struct.SC, ptr [[VAR111:%.+]], i{{.+}} 0, i{{.+}} 2
// CK24-DAG: [[VAR111]] = load ptr, ptr %p
// CK24-DAG: [[SEC1]] = getelementptr {{.*}}%struct.SB, ptr [[SEC11:%[^,]+]], i{{.+}} 0
// CK24-DAG: [[SEC11]] = load ptr, ptr [[SEC111:%[^,]+]],
// CK24-DAG: [[SEC111]] = getelementptr {{.*}}%struct.SC, ptr [[SEC1111:%.+]], i{{.+}} 0, i{{.+}} 2
// CK24-DAG: [[SEC1111]] = load ptr, ptr %p

// CK24-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 2
// CK24-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 2
// CK24-DAG: store ptr [[VAR11]], ptr [[BP2]]
// CK24-DAG: store ptr [[SEC1]], ptr [[P2]]

// CK24: call void [[CALL17:@.+]](ptr {{[^,]+}})
#pragma omp target map(p->p[:5])
  { p->a++; }

// Region 18

//  &p[0], &p->s.sa[3].a,           sizeof(p->s.sa[3].a), TO | FROM | PARAM
//  &p,    &p->s.sa[3].a sizeof(p), ATTACH

// CK24-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK24-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK24-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK24-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK24-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CK24-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK24-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK24-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: store ptr [[VAR0:%.+]], ptr [[BP0]]
// CK24-DAG: store ptr [[SEC0:%.+]], ptr [[P0]]
// CK24-DAG: [[SEC0]] = getelementptr {{.*}}ptr [[SEC00:%[^,]+]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[SEC00]] = getelementptr {{.*}}ptr [[SEC000:%[^,]+]], i{{.+}} 0, i{{.+}} 3
// CK24-DAG: [[SEC000]] = getelementptr {{.*}}ptr [[SEC0000:%[^,]+]], i{{.+}} 0, i{{.+}} 2
// CK24-DAG: [[SEC0000]] = getelementptr {{.*}}ptr [[VAR00:%.+]], i{{.+}} 0, i{{.+}} 1

// CK24-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: store ptr [[VAR:%[^,]+]], ptr [[BP1]]
// CK24-DAG: store ptr [[SEC1:%.+]], ptr [[P1]]
// CK24-DAG: [[SEC1]] = getelementptr {{.*}}ptr [[SEC11:%[^,]+]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[SEC11]] = getelementptr {{.*}}ptr [[SEC111:%[^,]+]], i{{.+}} 0, i{{.+}} 3
// CK24-DAG: [[SEC111]] = getelementptr {{.*}}ptr [[SEC1111:%[^,]+]], i{{.+}} 0, i{{.+}} 2
// CK24-DAG: [[SEC1111]] = getelementptr {{.*}}ptr [[VAR11:%.+]], i{{.+}} 0, i{{.+}} 1

// CK24-DAG: [[VAR0]] = load ptr, ptr [[VAR]]
// CK24-DAG: [[VAR00]] = load ptr, ptr [[VAR]]
// CK24-DAG: [[VAR11]] = load ptr, ptr [[VAR]]

// CK24: call void [[CALL18:@.+]](ptr {{[^,]+}})
#pragma omp target map(p->s.sa[3].a)
  { p->a++; }

// Region 19

//  &p[0],          &p[0],          0,                     IMPLICIT | PARAM
//  &p->s.sp[3][0], &p->s.sp[3]->a, sizeof(p->s.sp[3]->a), TO | FROM
//  &p->s.sp[3],    &p->s.sp[3]->a, sizeof(p),             ATTACH

// CK24-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK24-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK24-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK24-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK24-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CK24-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK24-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK24-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: store ptr [[VAR0:%.+]], ptr [[BP0]]
// CK24-DAG: store ptr [[VAR0]], ptr [[P0]]
// CK24-DAG: [[VAR0]] = load ptr, ptr %p

// CK24-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: store ptr [[VAR1:%.+]], ptr [[BP1]]
// CK24-DAG: store ptr [[SEC1:%.+]], ptr [[P1]]
// CK24-DAG: [[VAR1]] = load ptr, ptr [[VAR11:%[^,]+]]
// CK24-DAG: [[VAR11]] = getelementptr {{.*}}[10 x ptr], ptr [[VAR111:%[^,]+]], i{{.+}} 0, i{{.+}} 3
// CK24-DAG: [[VAR111]] = getelementptr {{.*}}%struct.SB, ptr [[VAR1111:%[^,]+]], i{{.+}} 0, i{{.+}} 3
// CK24-DAG: [[VAR1111]] = getelementptr {{.*}}%struct.SC, ptr [[VAR11111:%.+]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[VAR11111]] = load ptr, ptr %p

// CK24-DAG: [[SEC1]] = getelementptr {{.*}}%struct.SA, ptr [[SEC11:%.+]], i{{.*}} 0, i{{.*}} 0
// CK24-DAG: [[SEC11]] = load ptr, ptr [[SEC111:%[^,]+]]
// CK24-DAG: [[SEC111]] = getelementptr {{.*}}[10 x ptr], ptr [[SEC1111:%[^,]+]], i{{.+}} 0, i{{.+}} 3
// CK24-DAG: [[SEC1111]] = getelementptr {{.*}}%struct.SB, ptr [[SEC11111:%[^,]+]], i{{.+}} 0, i{{.+}} 3
// CK24-DAG: [[SEC11111]] = getelementptr {{.*}}%struct.SC, ptr [[SEC111111:%.+]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[SEC111111]] = load ptr, ptr %p

// CK24-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 2
// CK24-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 2
// CK24-DAG: store ptr [[VAR11]], ptr [[BP2]]
// CK24-DAG: store ptr [[SEC1]], ptr [[P2]]

// CK24: call void [[CALL19:@.+]](ptr {{[^,]+}})
#pragma omp target map(p->s.sp[3]->a)
  { p->a++; }

// Region 20

//  &p[0],    &p[0],    0,               IMPLICIT | PARAM
//  &p->p[0], &p->p->a, sizeof(p->p->a), TO | FROM
//  &p->p,    &p->p->a, sizeof(p),       ATTACH

// CK24-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK24-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK24-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK24-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK24-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CK24-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK24-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK24-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: store ptr [[VAR0:%.+]], ptr [[BP0]]
// CK24-DAG: store ptr [[VAR0]], ptr [[P0]]
// CK24-DAG: [[VAR0]] = load ptr, ptr %p

// CK24-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: store ptr [[VAR1:%.+]], ptr [[BP1]]
// CK24-DAG: store ptr [[SEC1:%.+]], ptr [[P1]]
// CK24-DAG: [[VAR1]] = load ptr, ptr [[VAR11:%[^,]+]]
// CK24-DAG: [[VAR11]] = getelementptr {{.*}}%struct.SC, ptr [[VAR111:.+]], i{{.*}} 0, i{{.*}} 2
// CK24-DAG: [[VAR111]] = load ptr, ptr %p

// CK24-DAG: [[SEC1]] = getelementptr {{.*}}%struct.SB, ptr [[SEC11:%[^,]+]], i{{.+}} 0
// CK24-DAG: [[SEC11]] = load ptr, ptr [[SEC111:%[^,]+]]
// CK24-DAG: [[SEC111]] = getelementptr {{.*}}%struct.SC, ptr [[SEC1111:.+]], i{{.*}} 0, i{{.*}} 2
// CK24-DAG: [[SEC1111]] = load ptr, ptr %p

// CK24-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 2
// CK24-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 2
// CK24-DAG: store ptr [[VAR11]], ptr [[BP2]]
// CK24-DAG: store ptr [[SEC1]], ptr [[P2]]

// CK24: call void [[CALL20:@.+]](ptr {{[^,]+}})
#pragma omp target map(p->p->a)
  { p->a++; }

// Region 21

//  &p[0],      &p[0],      0,                 IMPLICIT | PARAM
//  &p->s.p[0], &p->s.p->a, sizeof(p->s.p->a), TO | FROM
//  &p->s.p,    &p->s.p->a, sizeof(p),         ATTACH

// CK24-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK24-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK24-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK24-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK24-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CK24-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK24-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK24-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: store ptr [[VAR0:%.+]], ptr [[BP0]]
// CK24-DAG: store ptr [[VAR0]], ptr [[P0]]
// CK24-DAG: [[VAR0]] = load ptr, ptr %p

// CK24-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: store ptr [[VAR1:%.+]], ptr [[BP1]]
// CK24-DAG: store ptr [[SEC1:%.+]], ptr [[P1]]
// CK24-DAG: [[VAR1]] = load ptr, ptr [[VAR11:%[^,]+]]
// CK24-DAG: [[VAR11]] = getelementptr {{.*}}%struct.SB, ptr [[VAR111:[^,]+]], i{{.+}} 0, i{{.+}} 4
// CK24-DAG: [[VAR111]] = getelementptr {{.*}}%struct.SC, ptr [[VAR1111:%.+]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[VAR1111]] = load ptr, ptr %p
// CK24-DAG: [[SEC1]] = getelementptr {{.*}}%struct.SA, ptr [[SEC11:%[^,]+]], i{{.+}} 0
// CK24-DAG: [[SEC11]] = load ptr, ptr [[SEC111:%[^,]+]]
// CK24-DAG: [[SEC111]] = getelementptr {{.*}}%struct.SB, ptr [[SEC1111:[^,]+]], i{{.+}} 0, i{{.+}} 4
// CK24-DAG: [[SEC1111]] = getelementptr {{.*}}%struct.SC, ptr [[SEC11111:%.+]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[SEC11111]] = load ptr, ptr %p

// CK24-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 2
// CK24-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 2
// CK24-DAG: store ptr [[VAR11]], ptr [[BP2]]
// CK24-DAG: store ptr [[SEC1]], ptr [[P2]]

// CK24: call void [[CALL21:@.+]](ptr {{[^,]+}})
#pragma omp target map(p->s.p->a)
  { p->a++; }

// Region 22

//  &p[0], &p->s.s.b[0], sizeof(p->s.s.b[0:2]), ALLOC | PARAM
//  &p[0], &p->s.s.b[0], sizeof(p->s.s.b[0:2]), TO | FROM
//  &p,    &p->s.s.b[0], sizeof(p),             ATTACH

// CK24-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK24-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK24-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK24-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK24-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CK24-DAG: [[SARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 4
// CK24-DAG: store ptr [[SIZES:%.+]], ptr [[SARG]]
// CK24-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK24-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
// CK24-DAG: [[SIZES]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

// CK24-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: store ptr [[VAR0:%.+]], ptr [[BP0]]
// CK24-DAG: store ptr [[SEC0:%.+]], ptr [[P0]]
// CK24-DAG: store i64 {{%.+}}, ptr [[S0]]
// CK24-DAG: [[SEC0]] = getelementptr {{.*}}ptr [[SEC00:%[^,]+]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[SEC00]] = getelementptr {{.*}}ptr [[SEC000:%[^,]+]], i{{.+}} 0, i{{.+}} 2
// CK24-DAG: [[SEC000]] = getelementptr {{.*}}ptr [[SEC0000:%[^,]+]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[SEC0000]] = getelementptr {{.*}}ptr [[VAR00:%.+]], i{{.+}} 0, i{{.+}} 1

// CK24-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: store ptr [[VAR0]], ptr [[BP1]]
// CK24-DAG: store ptr [[SEC0]], ptr [[P1]]

// CK24-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 2
// CK24-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 2
// CK24-DAG: store ptr [[VAR:%[^,]+]], ptr [[BP2]]
// CK24-DAG: store ptr [[SEC0]], ptr [[P2]]

// CK24-DAG: [[VAR0]] = load ptr, ptr [[VAR]]
// CK24-DAG: [[VAR00]] = load ptr, ptr [[VAR]]

// CK24: call void [[CALL22:@.+]](ptr {{[^,]+}})
#pragma omp target map(p->s.s.b[:2])
  { p->a++; }

// Region 23

//  &p[0],      &p[0],         0,                     IMPLICIT | PARAM
//  &p->s.p[0], &p->s.p->b[0], sizeof(p->s.p->b[:2]), ALLOC
//  &p->s.p[0], &p->s.p->b[0], sizeof(p->s.p->b[:2]), MEMBER_OF_1 | TO | FROM
//  &p->s.p,    &p->s.p->b[0], sizeof(p),             ATTACH

// CK24-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK24-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK24-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK24-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK24-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CK24-DAG: [[SARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 4
// CK24-DAG: store ptr [[SIZES:%.+]], ptr [[SARG]]
// CK24-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK24-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
// CK24-DAG: [[SIZES]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

// CK24-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: store ptr [[VAR0:%.+]], ptr [[BP0]]
// CK24-DAG: store ptr [[VAR0]], ptr [[P0]]
// CK24-DAG: [[VAR0]] = load ptr, ptr %p

// CK24-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[S1:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: store ptr [[VAR1:%.+]], ptr [[BP1]]
// CK24-DAG: store ptr [[SEC1:%.+]], ptr [[P1]]
// CK24-DAG: store i64 {{%.+}}, ptr [[S1]]
// CK24-DAG: [[VAR1]] = load ptr, ptr [[VAR11:%[^,]+]]
// CK24-DAG: [[VAR11]] = getelementptr {{.*}}%struct.SB, ptr [[VAR111:%.+]], i{{.*}} 0, i{{.*}} 4
// CK24-DAG: [[VAR111]] = getelementptr {{.*}}%struct.SC, ptr [[VAR1111:%.+]], i{{.*}} 0, i{{.*}} 1
// CK24-DAG: [[VAR1111]] = load ptr, ptr %p

// CK24-DAG: [[SEC1]] = getelementptr {{.*}}[10 x i{{.*}}], ptr [[SEC11:%.+]], i{{.*}} 0, i{{.*}} 0
// CK24-DAG: [[SEC11]] = getelementptr {{.*}}%struct.SA, ptr [[SEC111:%.+]], i{{.*}} 0, i{{.*}} 2
// CK24-DAG: [[SEC111]] = load ptr, ptr [[SEC1111:%[^,]+]]
// CK24-DAG: [[SEC1111]] = getelementptr {{.*}}%struct.SB, ptr [[SEC11111:%.+]], i{{.*}} 0, i{{.*}} 4
// CK24-DAG: [[SEC11111]] = getelementptr {{.*}}%struct.SC, ptr [[SEC111111:%.+]], i{{.*}} 0, i{{.*}} 1
// CK24-DAG: [[SEC111111]] = load ptr, ptr %p

// CK24-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 2
// CK24-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 2
// CK24-DAG: store ptr [[VAR1]], ptr [[BP2]]
// CK24-DAG: store ptr [[SEC1]], ptr [[P2]]

// CK24-DAG: [[BP3:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 3
// CK24-DAG: [[P3:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 3
// CK24-DAG: store ptr [[VAR11]], ptr [[BP3]]
// CK24-DAG: store ptr [[SEC1]], ptr [[P3]]

// CK24: call void [[CALL23:@.+]](ptr {{[^,]+}})
#pragma omp target map(p->s.p->b[:2])
  { p->a++; }

// Region 24

//  &p[0],          &p[0],          0,                     IMPLICIT | PARAM
//  &p->p->p->p[0], &p->p->p->p->a, sizeof(p->p->p->p->a), TO | FROM
//  &p->p->p->p,    &p->p->p->p->a, sizeof(p),             ATTACH

// CK24-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK24-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK24-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK24-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK24-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CK24-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK24-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK24-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: store ptr [[VAR0:%.+]], ptr [[BP0]]
// CK24-DAG: store ptr [[VAR0]], ptr [[P0]]
// CK24-DAG: [[VAR0]] = load ptr, ptr %p

// CK24-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: store ptr [[VAR1:%.+]], ptr [[BP1]]
// CK24-DAG: store ptr [[SEC1:%.+]], ptr [[P1]]
// CK24-DAG: [[VAR1]] = load ptr, ptr [[VAR11:%[^,]+]]
// CK24-DAG: [[VAR11]] = getelementptr {{.*}}%struct.SA, ptr [[VAR111:%.+]], i{{.*}} 0, i{{.*}} 1
// CK24-DAG: [[VAR111]] = load ptr, ptr [[VAR1111:%[^,]+]]
// CK24-DAG: [[VAR1111]] = getelementptr {{.*}}%struct.SB, ptr [[VAR11111:%.+]], i{{.*}} 0, i{{.*}} 4
// CK24-DAG: [[VAR11111]] = load ptr, ptr [[VAR111111:%[^,]+]]
// CK24-DAG: [[VAR111111]] = getelementptr {{.*}}%struct.SC, ptr [[VAR1111111:%.+]], i{{.*}} 0, i{{.*}} 2
// CK24-DAG: [[VAR1111111]] = load ptr, ptr %p

// CK24-DAG: [[SEC1:%.+]] = getelementptr {{.*}}%struct.SA, ptr [[SEC11:%.+]], i{{.*}} 0, i{{.*}} 0
// CK24-DAG: [[SEC11]] = load ptr, ptr [[SEC111:%[^,]+]]
// CK24-DAG: [[SEC111]] = getelementptr {{.*}}%struct.SA, ptr [[SEC1111:%.+]], i{{.*}} 0, i{{.*}} 1
// CK24-DAG: [[SEC1111]] = load ptr, ptr [[SEC11111:%[^,]+]]
// CK24-DAG: [[SEC11111]] = getelementptr {{.*}}%struct.SB, ptr [[SEC111111:%.+]], i{{.*}} 0, i{{.*}} 4
// CK24-DAG: [[SEC111111]] = load ptr, ptr [[SEC1111111:%[^,]+]]
// CK24-DAG: [[SEC1111111]] = getelementptr {{.*}}%struct.SC, ptr [[SEC11111111:%.+]], i{{.*}} 0, i{{.*}} 2
// CK24-DAG: [[SEC11111111]] = load ptr, ptr %p

// CK24-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 2
// CK24-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 2
// CK24-DAG: store ptr [[VAR11]], ptr [[BP2]]
// CK24-DAG: store ptr [[SEC1]], ptr [[P2]]

// CK24: call void [[CALL24:@.+]](ptr {{[^,]+}})
#pragma omp target map(p->p->p->p->a)
  { p->a++; }

  return s.a;
}

// CK24: define {{.+}}[[CALL01]]
// CK24: define {{.+}}[[CALL13]]
// CK24: define {{.+}}[[CALL14]]
// CK24: define {{.+}}[[CALL15]]
// CK24: define {{.+}}[[CALL16]]
// CK24: define {{.+}}[[CALL17]]
// CK24: define {{.+}}[[CALL18]]
// CK24: define {{.+}}[[CALL19]]
// CK24: define {{.+}}[[CALL20]]
// CK24: define {{.+}}[[CALL21]]
// CK24: define {{.+}}[[CALL22]]
// CK24: define {{.+}}[[CALL23]]
// CK24: define {{.+}}[[CALL24]]
#endif // CK24
#endif
