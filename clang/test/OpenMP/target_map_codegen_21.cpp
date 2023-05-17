// expected-no-diagnostics
#ifndef HEADER
#define HEADER

///==========================================================================///
// RUN: %clang_cc1 -DCK22 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK22 --check-prefix CK22-64
// RUN: %clang_cc1 -DCK22 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK22 --check-prefix CK22-64
// RUN: %clang_cc1 -DCK22 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK22 --check-prefix CK22-32
// RUN: %clang_cc1 -DCK22 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK22 --check-prefix CK22-32

// RUN: %clang_cc1 -DCK22 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK22 --check-prefix CK22-64
// RUN: %clang_cc1 -DCK22 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK22 --check-prefix CK22-64
// RUN: %clang_cc1 -DCK22 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK22 --check-prefix CK22-32
// RUN: %clang_cc1 -DCK22 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK22 --check-prefix CK22-32

// RUN: %clang_cc1 -DCK22 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK22 --check-prefix CK22-64
// RUN: %clang_cc1 -DCK22 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK22 --check-prefix CK22-64
// RUN: %clang_cc1 -DCK22 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK22 --check-prefix CK22-32
// RUN: %clang_cc1 -DCK22 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK22 --check-prefix CK22-32

// RUN: %clang_cc1 -DCK22 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY21 %s
// RUN: %clang_cc1 -DCK22 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY21 %s
// RUN: %clang_cc1 -DCK22 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY21 %s
// RUN: %clang_cc1 -DCK22 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY21 %s
// SIMD-ONLY21-NOT: {{__kmpc|__tgt}}
#ifdef CK22

// CK22-DAG: [[ST:%.+]] = type { float }
// CK22-DAG: [[STT:%.+]] = type { i32 }

// CK22-LABEL: @.__omp_offloading_{{.*}}explicit_maps_globals{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK22: [[SIZE00:@.+]] = private {{.*}}constant [1 x i64] [i64 4]
// CK22: [[MTYPE00:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK22-LABEL: @.__omp_offloading_{{.*}}explicit_maps_globals{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK22: [[SIZE01:@.+]] = private {{.*}}constant [1 x i64] [i64 400]
// CK22: [[MTYPE01:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK22-LABEL: @.__omp_offloading_{{.*}}explicit_maps_globals{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK22: [[SIZE02:@.+]] = private {{.*}}constant [1 x i64] [i64 {{8|4}}]
// CK22: [[MTYPE02:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK22-LABEL: @.__omp_offloading_{{.*}}explicit_maps_globals{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK22: [[SIZE03:@.+]] = private {{.*}}constant [1 x i64] [i64 16]
// CK22: [[MTYPE03:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK22-LABEL: @.__omp_offloading_{{.*}}explicit_maps_globals{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK22: [[SIZE04:@.+]] = private {{.*}}constant [1 x i64] [i64 20]
// CK22: [[MTYPE04:@.+]] = private {{.*}}constant [1 x i64] [i64 51]

// CK22-LABEL: @.__omp_offloading_{{.*}}explicit_maps_globals{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK22: [[SIZE05:@.+]] = private {{.*}}constant [1 x i64] [i64 4]
// CK22: [[MTYPE05:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK22-LABEL: @.__omp_offloading_{{.*}}explicit_maps_globals{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK22: [[SIZE06:@.+]] = private {{.*}}constant [1 x i64] [i64 400]
// CK22: [[MTYPE06:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK22-LABEL: @.__omp_offloading_{{.*}}explicit_maps_globals{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK22: [[SIZE07:@.+]] = private {{.*}}constant [1 x i64] [i64 {{8|4}}]
// CK22: [[MTYPE07:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK22-LABEL: @.__omp_offloading_{{.*}}explicit_maps_globals{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK22: [[SIZE08:@.+]] = private {{.*}}constant [1 x i64] [i64 16]
// CK22: [[MTYPE08:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK22-LABEL: @.__omp_offloading_{{.*}}explicit_maps_globals{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK22: [[SIZE09:@.+]] = private {{.*}}constant [1 x i64] [i64 20]
// CK22: [[MTYPE09:@.+]] = private {{.*}}constant [1 x i64] [i64 51]

// CK22-LABEL: @.__omp_offloading_{{.*}}explicit_maps_globals{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK22: [[SIZE10:@.+]] = private {{.*}}constant [1 x i64] [i64 4]
// CK22: [[MTYPE10:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK22-LABEL: @.__omp_offloading_{{.*}}explicit_maps_globals{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK22: [[SIZE11:@.+]] = private {{.*}}constant [1 x i64] [i64 400]
// CK22: [[MTYPE11:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK22-LABEL: @.__omp_offloading_{{.*}}explicit_maps_globals{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK22: [[SIZE12:@.+]] = private {{.*}}constant [1 x i64] [i64 {{8|4}}]
// CK22: [[MTYPE12:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK22-LABEL: @.__omp_offloading_{{.*}}explicit_maps_globals{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK22: [[SIZE13:@.+]] = private {{.*}}constant [1 x i64] [i64 16]
// CK22: [[MTYPE13:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK22-LABEL: @.__omp_offloading_{{.*}}explicit_maps_globals{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK22: [[SIZE14:@.+]] = private {{.*}}constant [1 x i64] [i64 20]
// CK22: [[MTYPE14:@.+]] = private {{.*}}constant [1 x i64] [i64 51]

int a;
int c[100];
int *d;

struct ST {
  float fa;
};

ST sa ;
ST sc[100];
ST *sd;

template<typename T>
struct STT {
  T fa;
};

STT<int> sta ;
STT<int> stc[100];
STT<int> *std;

// CK22-LABEL: explicit_maps_globals{{.*}}(
int explicit_maps_globals(void){
// Region 00
// CK22-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK22-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK22-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK22-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK22-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CK22-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK22-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK22-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK22-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK22-DAG: store ptr @a, ptr [[BP0]]
// CK22-DAG: store ptr @a, ptr [[P0]]

// CK22: call void [[CALL00:@.+]](ptr {{[^,]+}})
#pragma omp target map(a)
  { a+=1; }

// Region 01
// CK22-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK22-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK22-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK22-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK22-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CK22-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK22-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK22-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK22-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK22-DAG: store ptr @c, ptr [[BP0]]
// CK22-DAG: store ptr @c, ptr [[P0]]

// CK22: call void [[CALL01:@.+]](ptr {{[^,]+}})
#pragma omp target map(c)
  { c[3]+=1; }

// Region 02
// CK22-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK22-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK22-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK22-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK22-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CK22-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK22-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK22-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK22-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK22-DAG: store ptr @d, ptr [[BP0]]
// CK22-DAG: store ptr @d, ptr [[P0]]

// CK22: call void [[CALL02:@.+]](ptr {{[^,]+}})
#pragma omp target map(d)
  { d[3]+=1; }

// Region 03
// CK22-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK22-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK22-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK22-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK22-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CK22-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK22-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK22-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK22-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK22-DAG: store ptr @c, ptr [[BP0]]
// CK22-DAG: store ptr getelementptr inbounds ([100 x i32], ptr @c, i{{.+}} 0, i{{.+}} 1), ptr [[P0]]

// CK22: call void [[CALL03:@.+]](ptr {{[^,]+}})
#pragma omp target map(c [1:4])
  { c[3]+=1; }

// Region 04
// CK22-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK22-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK22-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK22-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK22-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CK22-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK22-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK22-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK22-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK22-DAG: store ptr @d, ptr [[BP0]]
// CK22-DAG: store ptr [[SEC0:%.+]], ptr [[P0]]
// CK22-DAG: [[SEC0]] = getelementptr {{.*}}ptr [[RVAR00:%.+]], i{{.+}} 2
// CK22-DAG: [[RVAR00]] = load ptr, ptr @d

// CK22: call void [[CALL04:@.+]](ptr {{[^,]+}})
#pragma omp target map(d [2:5])
  { d[3]+=1; }

// Region 05
// CK22-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK22-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK22-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK22-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK22-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CK22-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK22-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK22-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK22-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK22-DAG: store ptr @sa, ptr [[BP0]]
// CK22-DAG: store ptr @sa, ptr [[P0]]

// CK22: call void [[CALL05:@.+]](ptr {{[^,]+}})
#pragma omp target map(sa)
  { sa.fa+=1; }

// Region 06
// CK22-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK22-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK22-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK22-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK22-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CK22-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK22-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK22-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK22-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK22-DAG: store ptr @sc, ptr [[BP0]]
// CK22-DAG: store ptr @sc, ptr [[P0]]

// CK22: call void [[CALL06:@.+]](ptr {{[^,]+}})
#pragma omp target map(sc)
  { sc[3].fa+=1; }

// Region 07
// CK22-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK22-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK22-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK22-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK22-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CK22-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK22-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK22-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK22-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK22-DAG: store ptr @sd, ptr [[BP0]]
// CK22-DAG: store ptr @sd, ptr [[P0]]

// CK22: call void [[CALL07:@.+]](ptr {{[^,]+}})
#pragma omp target map(sd)
  { sd[3].fa+=1; }

// Region 08
// CK22-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK22-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK22-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK22-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK22-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CK22-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK22-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK22-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK22-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK22-DAG: store ptr @sc, ptr [[BP0]]
// CK22-DAG: store ptr getelementptr inbounds ([100 x [[ST]]], ptr @sc, i{{.+}} 0, i{{.+}} 1), ptr [[P0]]

// CK22: call void [[CALL08:@.+]](ptr {{[^,]+}})
#pragma omp target map(sc [1:4])
  { sc[3].fa+=1; }

// Region 09
// CK22-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK22-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK22-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK22-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK22-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CK22-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK22-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK22-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK22-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK22-DAG: store ptr @sd, ptr [[BP0]]
// CK22-DAG: store ptr [[SEC0:%.+]], ptr [[P0]]
// CK22-DAG: [[SEC0]] = getelementptr {{.*}}ptr [[RVAR00:%.+]], i{{.+}} 2
// CK22-DAG: [[RVAR00]] = load ptr, ptr @sd

// CK22: call void [[CALL09:@.+]](ptr {{[^,]+}})
#pragma omp target map(sd [2:5])
  { sd[3].fa+=1; }

// Region 10
// CK22-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK22-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK22-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK22-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK22-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CK22-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK22-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK22-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK22-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK22-DAG: store ptr @sta, ptr [[BP0]]
// CK22-DAG: store ptr @sta, ptr [[P0]]

// CK22: call void [[CALL10:@.+]](ptr {{[^,]+}})
#pragma omp target map(sta)
  { sta.fa+=1; }

// Region 11
// CK22-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK22-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK22-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK22-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK22-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CK22-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK22-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK22-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK22-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK22-DAG: store ptr @stc, ptr [[BP0]]
// CK22-DAG: store ptr @stc, ptr [[P0]]

// CK22: call void [[CALL11:@.+]](ptr {{[^,]+}})
#pragma omp target map(stc)
  { stc[3].fa+=1; }

// Region 12
// CK22-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK22-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK22-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK22-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK22-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CK22-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK22-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK22-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK22-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK22-DAG: store ptr @std, ptr [[BP0]]
// CK22-DAG: store ptr @std, ptr [[P0]]

// CK22: call void [[CALL12:@.+]](ptr {{[^,]+}})
#pragma omp target map(std)
  { std[3].fa+=1; }

// Region 13
// CK22-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK22-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK22-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK22-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK22-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CK22-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK22-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK22-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK22-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK22-DAG: store ptr @stc, ptr [[BP0]]
// CK22-DAG: store ptr getelementptr inbounds ([100 x [[STT]]], ptr @stc, i{{.+}} 0, i{{.+}} 1),  ptr [[P0]]

// CK22: call void [[CALL13:@.+]](ptr {{[^,]+}})
#pragma omp target map(stc [1:4])
  { stc[3].fa+=1; }

// Region 14
// CK22-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK22-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK22-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK22-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK22-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CK22-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK22-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK22-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK22-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK22-DAG: store ptr @std, ptr [[BP0]]
// CK22-DAG: store ptr [[SEC0:%.+]], ptr [[P0]]
// CK22-DAG: [[SEC0]] = getelementptr {{.*}}ptr [[RVAR00:%.+]], i{{.+}} 2
// CK22-DAG: [[RVAR00]] = load ptr, ptr @std

// CK22: call void [[CALL14:@.+]](ptr {{[^,]+}})
#pragma omp target map(std [2:5])
  { std[3].fa+=1; }

  return 0;
}
// CK22: define {{.+}}[[CALL00]]
// CK22: define {{.+}}[[CALL01]]
// CK22: define {{.+}}[[CALL02]]
// CK22: define {{.+}}[[CALL03]]
// CK22: define {{.+}}[[CALL04]]
// CK22: define {{.+}}[[CALL05]]
// CK22: define {{.+}}[[CALL06]]
// CK22: define {{.+}}[[CALL07]]
// CK22: define {{.+}}[[CALL08]]
// CK22: define {{.+}}[[CALL09]]
// CK22: define {{.+}}[[CALL10]]
// CK22: define {{.+}}[[CALL11]]
// CK22: define {{.+}}[[CALL12]]
// CK22: define {{.+}}[[CALL13]]
// CK22: define {{.+}}[[CALL14]]
#endif // CK22
#endif
