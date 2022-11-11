// expected-no-diagnostics
#ifndef HEADER
#define HEADER

///==========================================================================///
// RUN: %clang_cc1 -DCK20 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK20 --check-prefix CK20-64
// RUN: %clang_cc1 -DCK20 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK20 --check-prefix CK20-64
// RUN: %clang_cc1 -DCK20 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK20 --check-prefix CK20-32
// RUN: %clang_cc1 -DCK20 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK20 --check-prefix CK20-32

// RUN: %clang_cc1 -DCK20 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK20 --check-prefix CK20-64
// RUN: %clang_cc1 -DCK20 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK20 --check-prefix CK20-64
// RUN: %clang_cc1 -DCK20 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK20 --check-prefix CK20-32
// RUN: %clang_cc1 -DCK20 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK20 --check-prefix CK20-32

// RUN: %clang_cc1 -DCK20 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK20 --check-prefix CK20-64
// RUN: %clang_cc1 -DCK20 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK20 --check-prefix CK20-64
// RUN: %clang_cc1 -DCK20 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK20 --check-prefix CK20-32
// RUN: %clang_cc1 -DCK20 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK20 --check-prefix CK20-32

// RUN: %clang_cc1 -DCK20 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY19 %s
// RUN: %clang_cc1 -DCK20 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY19 %s
// RUN: %clang_cc1 -DCK20 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY19 %s
// RUN: %clang_cc1 -DCK20 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY19 %s
// SIMD-ONLY19-NOT: {{__kmpc|__tgt}}
#ifdef CK20

// CK20-LABEL: @.__omp_offloading_{{.*}}explicit_maps_references_and_function_args{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK20: [[SIZE00:@.+]] = private {{.*}}constant [1 x i64] [i64 4]
// CK20: [[MTYPE00:@.+]] = private {{.*}}constant [1 x i64] [i64 33]

// CK20-LABEL: @.__omp_offloading_{{.*}}explicit_maps_references_and_function_args{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK20: [[SIZE01:@.+]] = private {{.*}}constant [1 x i64] [i64 20]
// CK20: [[MTYPE01:@.+]] = private {{.*}}constant [1 x i64] [i64 33]

// CK20-LABEL: @.__omp_offloading_{{.*}}explicit_maps_references_and_function_args{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK20: [[SIZE02:@.+]] = private {{.*}}constant [1 x i64] [i64 4]
// CK20: [[MTYPE02:@.+]] = private {{.*}}constant [1 x i64] [i64 34]

// CK20-LABEL: @.__omp_offloading_{{.*}}explicit_maps_references_and_function_args{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK20: [[SIZE03:@.+]] = private {{.*}}constant [1 x i64] [i64 12]
// CK20: [[MTYPE03:@.+]] = private {{.*}}constant [1 x i64] [i64 34]

// CK20-LABEL: explicit_maps_references_and_function_args{{.*}}(
void explicit_maps_references_and_function_args (int a, float b, int (&c)[10], float *d){

  int &aa = a;
  float &bb = b;
  int (&cc)[10] = c;
  float *&dd = d;

// Region 00
// CK20-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK20-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK20-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK20-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK20-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CK20-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK20-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK20-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK20-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK20-DAG: store ptr [[RVAR0:%.+]], ptr [[BP0]]
// CK20-DAG: store ptr [[RVAR00:%.+]], ptr [[P0]]
// CK20-DAG: [[RVAR0]] = load ptr, ptr [[VAR0:%[^,]+]]
// CK20-DAG: [[RVAR00]] = load ptr, ptr [[VAR0]]

// CK20: call void [[CALL00:@.+]](ptr {{[^,]+}})
#pragma omp target map(to \
                       : aa)
  {
    aa += 1;
  }

// Region 01
// CK20-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK20-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK20-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK20-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK20-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CK20-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK20-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK20-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK20-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK20-DAG: store ptr [[RVAR0:%.+]], ptr [[BP0]]
// CK20-DAG: store ptr [[SEC0:%.+]], ptr [[P0]]
// CK20-DAG: [[SEC0]] = getelementptr {{.*}}ptr [[RVAR00:%.+]], i{{.+}} 0, i{{.+}} 0
// CK20-DAG: [[RVAR0]] = load ptr, ptr [[VAR0:%[^,]+]]
// CK20-DAG: [[RVAR00]] = load ptr, ptr [[VAR0]]

// CK20: call void [[CALL01:@.+]](ptr {{[^,]+}})
#pragma omp target map(to \
                       : cc[:5])
  {
    cc[3] += 1;
  }

// Region 02
// CK20-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK20-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK20-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK20-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK20-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CK20-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK20-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK20-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK20-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK20-DAG: store ptr [[VAR0:%.+]], ptr [[BP0]]
// CK20-DAG: store ptr [[VAR0]], ptr [[P0]]

// CK20: call void [[CALL02:@.+]](ptr {{[^,]+}})
#pragma omp target map(from \
                       : b)
  {
    b += 1.0f;
  }

// Region 03
// CK20-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK20-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK20-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK20-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK20-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CK20-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK20-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK20-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK20-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK20-DAG: store ptr [[RVAR0:%.+]], ptr [[BP0]]
// CK20-DAG: store ptr [[SEC0:%.+]], ptr [[P0]]
// CK20-DAG: [[RVAR0]] = load ptr, ptr [[VAR0:%[^,]+]]
// CK20-DAG: [[SEC0]] = getelementptr {{.*}}ptr [[RVAR00:%.+]], i{{.+}} 2
// CK20-DAG: [[RVAR00]] = load ptr, ptr [[VAR0]]

// CK20: call void [[CALL03:@.+]](ptr {{[^,]+}})
#pragma omp target map(from \
                       : d [2:3])
  {
    d[2] += 1.0f;
  }
}

// CK20: define {{.+}}[[CALL00]]
// CK20: define {{.+}}[[CALL01]]
// CK20: define {{.+}}[[CALL02]]
// CK20: define {{.+}}[[CALL03]]

#endif // CK20
#endif
