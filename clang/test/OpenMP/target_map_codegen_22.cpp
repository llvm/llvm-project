// expected-no-diagnostics
#ifndef HEADER
#define HEADER

///==========================================================================///
// RUN: %clang_cc1 -std=c++11 -DCK23 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK23 --check-prefix CK23-64
// RUN: %clang_cc1 -std=c++11 -DCK23 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -std=c++11 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK23 --check-prefix CK23-64
// RUN: %clang_cc1 -std=c++11 -DCK23 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK23 --check-prefix CK23-32
// RUN: %clang_cc1 -std=c++11 -DCK23 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -std=c++11 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK23 --check-prefix CK23-32

// RUN: %clang_cc1 -std=c++11 -DCK23 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK23 --check-prefix CK23-64
// RUN: %clang_cc1 -std=c++11 -DCK23 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -std=c++11 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK23 --check-prefix CK23-64
// RUN: %clang_cc1 -std=c++11 -DCK23 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK23 --check-prefix CK23-32
// RUN: %clang_cc1 -std=c++11 -DCK23 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -std=c++11 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK23 --check-prefix CK23-32

// RUN: %clang_cc1 -std=c++11 -DCK23 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK23 --check-prefix CK23-64
// RUN: %clang_cc1 -std=c++11 -DCK23 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -std=c++11 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK23 --check-prefix CK23-64
// RUN: %clang_cc1 -std=c++11 -DCK23 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK23 --check-prefix CK23-32
// RUN: %clang_cc1 -std=c++11 -DCK23 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -std=c++11 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK23 --check-prefix CK23-32

// RUN: %clang_cc1 -std=c++11 -DCK23 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY22 %s
// RUN: %clang_cc1 -std=c++11 -DCK23 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -std=c++11 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY22 %s
// RUN: %clang_cc1 -std=c++11 -DCK23 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY22 %s
// RUN: %clang_cc1 -std=c++11 -DCK23 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -std=c++11 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY22 %s
// SIMD-ONLY22-NOT: {{__kmpc|__tgt}}
#ifdef CK23

// CK23-LABEL: @.__omp_offloading_{{.*}}explicit_maps_inside_captured{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK23: [[SIZE00:@.+]] = private {{.*}}constant [1 x i64] [i64 4]
// CK23: [[MTYPE00:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK23-LABEL: @.__omp_offloading_{{.*}}explicit_maps_inside_captured{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK23: [[SIZE01:@.+]] = private {{.*}}constant [1 x i64] [i64 4]
// CK23: [[MTYPE01:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK23-LABEL: @.__omp_offloading_{{.*}}explicit_maps_inside_captured{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK23: [[SIZE02:@.+]] = private {{.*}}constant [1 x i64] [i64 400]
// CK23: [[MTYPE02:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK23-LABEL: @.__omp_offloading_{{.*}}explicit_maps_inside_captured{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK23: [[SIZE03:@.+]] = private {{.*}}constant [1 x i64] [i64 {{8|4}}]
// CK23: [[MTYPE03:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK23-LABEL: @.__omp_offloading_{{.*}}explicit_maps_inside_captured{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK23: [[SIZE04:@.+]] = private {{.*}}constant [1 x i64] [i64 16]
// CK23: [[MTYPE04:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK23-LABEL: @.__omp_offloading_{{.*}}explicit_maps_inside_captured{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK23: [[SIZE05:@.+]] = private {{.*}}constant [1 x i64] [i64 16]
// CK23: [[MTYPE05:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK23-LABEL: explicit_maps_inside_captured{{.*}}(
int explicit_maps_inside_captured(int a){
  float b;
  float c[100];
  float *d;

  // CK23: call void @{{.*}}explicit_maps_inside_captured{{.*}}(ptr {{.*}})
  // CK23: define {{.*}}explicit_maps_inside_captured{{.*}}
  [&](void){
// Region 00
// CK23-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK23-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK23-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK23-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK23-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CK23-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK23-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK23-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK23-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK23-DAG: store ptr [[VAR0:%.+]], ptr [[BP0]]
// CK23-DAG: store ptr [[VAR00:%.+]], ptr [[P0]]
// CK23-DAG: [[VAR0]] = load ptr, ptr [[CAP0:%[^,]+]]
// CK23-DAG: [[CAP0]] = getelementptr inbounds nuw %class.anon,
// CK23-DAG: [[VAR00]] = load ptr, ptr [[CAP00:%[^,]+]]
// CK23-DAG: [[CAP00]] = getelementptr inbounds nuw %class.anon,

// CK23: call void [[CALL00:@.+]](ptr {{[^,]+}})
#pragma omp target map(a)
      { a+=1; }
// Region 01
// CK23-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK23-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK23-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK23-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK23-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CK23-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK23-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK23-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK23-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK23-DAG: store ptr [[VAR0:%.+]], ptr [[BP0]]
// CK23-DAG: store ptr [[VAR00:%.+]], ptr [[P0]]
// CK23-DAG: [[VAR0]] = load ptr, ptr [[CAP0:%[^,]+]]
// CK23-DAG: [[CAP0]] = getelementptr inbounds nuw %class.anon,
// CK23-DAG: [[VAR00]] = load ptr, ptr [[CAP00:%[^,]+]]
// CK23-DAG: [[CAP00]] = getelementptr inbounds nuw %class.anon,

// CK23: call void [[CALL01:@.+]](ptr {{[^,]+}})
#pragma omp target map(b)
      { b+=1; }
// Region 02
// CK23-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK23-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK23-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK23-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK23-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CK23-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK23-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK23-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK23-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK23-DAG: store ptr [[VAR0:%.+]], ptr [[BP0]]
// CK23-DAG: store ptr [[VAR00:%.+]], ptr [[P0]]
// CK23-DAG: [[VAR0]] = load ptr, ptr [[CAP0:%[^,]+]]
// CK23-DAG: [[CAP0]] = getelementptr inbounds nuw %class.anon,
// CK23-DAG: [[VAR00]] = load ptr, ptr [[CAP00:%[^,]+]]
// CK23-DAG: [[CAP00]] = getelementptr inbounds nuw %class.anon,

// CK23: call void [[CALL02:@.+]](ptr {{[^,]+}})
#pragma omp target map(c)
      { c[3]+=1; }

// Region 03
// CK23-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK23-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK23-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK23-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK23-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CK23-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK23-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK23-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK23-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK23-DAG: store ptr [[VAR0:%.+]], ptr [[BP0]]
// CK23-DAG: store ptr [[VAR00:%.+]], ptr [[P0]]
// CK23-DAG: [[VAR0]] = load ptr, ptr [[CAP0:%[^,]+]]
// CK23-DAG: [[CAP0]] = getelementptr inbounds nuw %class.anon,
// CK23-DAG: [[VAR00]] = load ptr, ptr [[CAP00:%[^,]+]]
// CK23-DAG: [[CAP00]] = getelementptr inbounds nuw %class.anon,

// CK23: call void [[CALL03:@.+]](ptr {{[^,]+}})
#pragma omp target map(d)
      { d[3]+=1; }
// Region 04
// CK23-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK23-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK23-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK23-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK23-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CK23-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK23-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK23-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK23-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK23-DAG: store ptr [[VAR0:%.+]], ptr [[BP0]]
// CK23-DAG: store ptr [[SEC0:%.+]], ptr [[P0]]
// CK23-DAG: [[SEC0]] = getelementptr {{.*}}ptr [[VAR00:%.+]], i{{.+}} 0, i{{.+}} 2
// CK23-DAG: [[VAR0]] = load ptr, ptr [[CAP0:%[^,]+]]
// CK23-DAG: [[CAP0]] = getelementptr inbounds nuw %class.anon,
// CK23-DAG: [[VAR00]] = load ptr, ptr [[CAP00:%[^,]+]]
// CK23-DAG: [[CAP00]] = getelementptr inbounds nuw %class.anon,

// CK23: call void [[CALL04:@.+]](ptr {{[^,]+}})
#pragma omp target map(c [2:4])
      { c[3]+=1; }

// Region 05
// CK23-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK23-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK23-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK23-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK23-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CK23-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK23-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK23-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK23-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK23-DAG: store ptr [[RVAR0:%.+]], ptr [[BP0]]
// CK23-DAG: store ptr [[SEC0:%.+]], ptr [[P0]]
// CK23-DAG: [[RVAR0]] = load ptr, ptr [[VAR0:%[^,]+]]
// CK23-DAG: [[SEC0]] = getelementptr {{.*}}ptr [[RVAR00:%.+]], i{{.+}} 2
// CK23-DAG: [[RVAR00]] = load ptr, ptr [[VAR00:%[^,]+]]
// CK23-DAG: [[VAR0]] = load ptr, ptr [[CAP0:%[^,]+]]
// CK23-DAG: [[CAP0]] = getelementptr inbounds nuw %class.anon,
// CK23-DAG: [[VAR00]] = load ptr, ptr [[CAP00:%[^,]+]]
// CK23-DAG: [[CAP00]] = getelementptr inbounds nuw %class.anon,

// CK23: call void [[CALL05:@.+]](ptr {{[^,]+}})
#pragma omp target map(d [2:4])
      { d[3]+=1; }
  }();
  return b;
}

// CK23: define {{.+}}[[CALL00]]
// CK23: define {{.+}}[[CALL01]]
// CK23: define {{.+}}[[CALL02]]
// CK23: define {{.+}}[[CALL03]]
// CK23: define {{.+}}[[CALL04]]
// CK23: define {{.+}}[[CALL05]]
#endif // CK23
#endif
