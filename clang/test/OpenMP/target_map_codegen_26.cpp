// expected-no-diagnostics
#ifndef HEADER
#define HEADER

///==========================================================================///
// RUN: %clang_cc1 -DCK27 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK27 --check-prefix CK27-64
// RUN: %clang_cc1 -DCK27 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK27 --check-prefix CK27-64
// RUN: %clang_cc1 -DCK27 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK27 --check-prefix CK27-32
// RUN: %clang_cc1 -DCK27 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK27 --check-prefix CK27-32

// RUN: %clang_cc1 -DCK27 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK27 --check-prefix CK27-64
// RUN: %clang_cc1 -DCK27 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK27 --check-prefix CK27-64
// RUN: %clang_cc1 -DCK27 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK27 --check-prefix CK27-32
// RUN: %clang_cc1 -DCK27 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK27 --check-prefix CK27-32

// RUN: %clang_cc1 -DCK27 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK27 --check-prefix CK27-64
// RUN: %clang_cc1 -DCK27 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK27 --check-prefix CK27-64
// RUN: %clang_cc1 -DCK27 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK27 --check-prefix CK27-32
// RUN: %clang_cc1 -DCK27 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK27 --check-prefix CK27-32

// RUN: %clang_cc1 -DCK27 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY26 %s
// RUN: %clang_cc1 -DCK27 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY26 %s
// RUN: %clang_cc1 -DCK27 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY26 %s
// RUN: %clang_cc1 -DCK27 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY26 %s
// SIMD-ONLY26-NOT: {{__kmpc|__tgt}}
#ifdef CK27

// CK27-LABEL: @.__omp_offloading_{{.*}}zero_size_section_and_private_maps{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK27: [[SIZE00:@.+]] = private {{.*}}constant [1 x i64] zeroinitializer
// CK27: [[MTYPE00:@.+]] = private {{.*}}constant [1 x i64] [i64 544]

// CK27-LABEL: @.__omp_offloading_{{.*}}zero_size_section_and_private_maps{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK27: [[SIZE01:@.+]] = private {{.*}}constant [1 x i64] zeroinitializer
// CK27: [[MTYPE01:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK27-LABEL: @.__omp_offloading_{{.*}}zero_size_section_and_private_maps{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK27: [[SIZE02:@.+]] = private {{.*}}constant [1 x i64] zeroinitializer
// CK27: [[MTYPE02:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK27-LABEL: @.__omp_offloading_{{.*}}zero_size_section_and_private_maps{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK27: [[SIZE03:@.+]] = private {{.*}}constant [1 x i64] zeroinitializer
// CK27: [[MTYPE03:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK27-LABEL: @.__omp_offloading_{{.*}}zero_size_section_and_private_maps{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK27-LABEL: @.__omp_offloading_{{.*}}zero_size_section_and_private_maps{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK27: [[SIZE05:@.+]] = private {{.*}}constant [1 x i64] zeroinitializer
// CK27: [[MTYPE05:@.+]] = private {{.*}}constant [1 x i64] [i64 32]

// CK27-LABEL: @.__omp_offloading_{{.*}}zero_size_section_and_private_maps{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK27-LABEL: @.__omp_offloading_{{.*}}zero_size_section_and_private_maps{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK27: [[SIZE07:@.+]] = private {{.*}}constant [1 x i64] [i64 4]
// CK27: [[MTYPE07:@.+]] = private {{.*}}constant [1 x i64] [i64 288]

// CK27-LABEL: @.__omp_offloading_{{.*}}zero_size_section_and_private_maps{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK27-LABEL: @.__omp_offloading_{{.*}}zero_size_section_and_private_maps{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK27: [[SIZE09:@.+]] = private {{.*}}constant [1 x i64] [i64 40]
// CK27: [[MTYPE09:@.+]] = private {{.*}}constant [1 x i64] [i64 161]

// CK27-LABEL: zero_size_section_and_private_maps{{.*}}(
void zero_size_section_and_private_maps (int ii){

  // Map of a pointer.
  int *pa;

// Region 00
// CK27-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK27-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK27-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK27-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK27-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CK27-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK27-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK27-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK27-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK27-DAG: store ptr [[VAR0:%.+]], ptr [[BP0]]
// CK27-DAG: store ptr [[VAR0]], ptr [[P0]]

// CK27: call void [[CALL00:@.+]](ptr {{[^,]+}})
#pragma omp target
  {
    pa[50]++;
  }

// Region 01
// CK27-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK27-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK27-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK27-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK27-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CK27-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK27-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK27-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK27-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK27-DAG: store ptr [[RVAR0:%.+]], ptr [[BP0]]
// CK27-DAG: store ptr [[SEC0:%.+]], ptr [[P0]]
// CK27-DAG: [[RVAR0]] = load ptr, ptr [[VAR0:%[^,]+]]
// CK27-DAG: [[SEC0]] = getelementptr {{.*}}ptr [[RVAR00:%.+]], i{{.+}} 0
// CK27-DAG: [[RVAR00]] = load ptr, ptr [[VAR0]]

// CK27: call void [[CALL01:@.+]](ptr {{[^,]+}})
#pragma omp target map(pa[:0])
  {
    pa[50]++;
  }

// Region 02
// CK27-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK27-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK27-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK27-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK27-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CK27-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK27-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK27-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK27-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK27-DAG: store ptr [[RVAR0:%.+]], ptr [[BP0]]
// CK27-DAG: store ptr [[SEC0:%.+]], ptr [[P0]]
// CK27-DAG: [[RVAR0]] = load ptr, ptr [[VAR0:%[^,]+]]
// CK27-DAG: [[SEC0]] = getelementptr {{.*}}ptr [[RVAR00:%.+]], i{{.+}} 0
// CK27-DAG: [[RVAR00]] = load ptr, ptr [[VAR0]]

// CK27: call void [[CALL02:@.+]](ptr {{[^,]+}})
#pragma omp target map(pa [0:0])
  {
    pa[50]++;
  }

// Region 03
// CK27-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK27-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK27-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK27-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK27-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CK27-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK27-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK27-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK27-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK27-DAG: store ptr [[RVAR0:%.+]], ptr [[BP0]]
// CK27-DAG: store ptr [[SEC0:%.+]], ptr [[P0]]
// CK27-DAG: [[RVAR0]] = load ptr, ptr [[VAR0:%[^,]+]]
// CK27-DAG: [[SEC0]] = getelementptr {{.*}}ptr [[RVAR00:%.+]], i{{.+}} %{{.+}}
// CK27-DAG: [[RVAR00]] = load ptr, ptr [[VAR0]]

// CK27: call void [[CALL03:@.+]](ptr {{[^,]+}})
#pragma omp target map(pa [ii:0])
  {
    pa[50]++;
  }

  int *pvtPtr;
  int pvtScl;
  int pvtArr[10];

// Region 04
// CK27-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK27: call void [[CALL04:@.+]]()
#pragma omp target private(pvtPtr)
  {
    pvtPtr[5]++;
  }

// Region 05
// CK27-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK27-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK27-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK27-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK27-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CK27-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK27-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK27-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK27-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK27-DAG: store ptr [[VAR0:%.+]], ptr [[BP0]]
// CK27-DAG: store ptr [[VAR0]], ptr [[P0]]

// CK27: call void [[CALL05:@.+]](ptr {{[^,]+}})
#pragma omp target firstprivate(pvtPtr)
  {
    pvtPtr[5]++;
  }

// Region 06
// CK27-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK27: call void [[CALL06:@.+]]()
#pragma omp target private(pvtScl)
  {
    pvtScl++;
  }

// Region 07
// CK27-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK27-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK27-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK27-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK27-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CK27-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
// CK27-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
// CK27-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
// CK27-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
// CK27-DAG: store i[[Z:64|32]] [[VAL:%.+]], ptr [[BP1]]
// CK27-DAG: store i[[Z]] [[VAL]], ptr [[P1]]
// CK27-DAG: [[VAL]] = load i[[Z]], ptr [[ADDR:%.+]],
// CK27-64-DAG: store i32 {{.+}}, ptr [[ADDR]],

// CK27: call void [[CALL07:@.+]](i[[Z]] [[VAL]])
#pragma omp target firstprivate(pvtScl)
  {
    pvtScl++;
  }

// Region 08
// CK27-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK27: call void [[CALL08:@.+]]()
#pragma omp target private(pvtArr)
  {
    pvtArr[5]++;
  }

// Region 09
// CK27-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK27-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK27-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK27-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK27-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CK27-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK27-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK27-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK27-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK27-DAG: store ptr [[VAR0:%.+]], ptr [[BP0]]
// CK27-DAG: store ptr [[VAR0]], ptr [[P0]]

// CK27: call void [[CALL09:@.+]](ptr {{[^,]+}})
#pragma omp target firstprivate(pvtArr)
  {
    pvtArr[5]++;
  }
}

// CK27: define {{.+}}[[CALL00]]
// CK27: define {{.+}}[[CALL01]]
// CK27: define {{.+}}[[CALL02]]
// CK27: define {{.+}}[[CALL03]]
// CK27: define {{.+}}[[CALL04]]
// CK27: define {{.+}}[[CALL05]]
// CK27: define {{.+}}[[CALL06]]
// CK27: define {{.+}}[[CALL07]]
#endif // CK27
#endif
