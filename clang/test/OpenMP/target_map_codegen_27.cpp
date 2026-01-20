// expected-no-diagnostics
#ifndef HEADER
#define HEADER

///==========================================================================///
// RUN: %clang_cc1 -DCK28 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK28 --check-prefix CK28-64
// RUN: %clang_cc1 -DCK28 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK28 --check-prefix CK28-64
// RUN: %clang_cc1 -DCK28 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK28 --check-prefix CK28-32
// RUN: %clang_cc1 -DCK28 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK28 --check-prefix CK28-32

// RUN: %clang_cc1 -DCK28 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK28 --check-prefix CK28-64
// RUN: %clang_cc1 -DCK28 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK28 --check-prefix CK28-64
// RUN: %clang_cc1 -DCK28 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK28 --check-prefix CK28-32
// RUN: %clang_cc1 -DCK28 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK28 --check-prefix CK28-32

// RUN: %clang_cc1 -DCK28 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK28 --check-prefix CK28-64
// RUN: %clang_cc1 -DCK28 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK28 --check-prefix CK28-64
// RUN: %clang_cc1 -DCK28 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK28 --check-prefix CK28-32
// RUN: %clang_cc1 -DCK28 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK28 --check-prefix CK28-32

// RUN: %clang_cc1 -DCK28 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY27 %s
// RUN: %clang_cc1 -DCK28 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY27 %s
// RUN: %clang_cc1 -DCK28 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY27 %s
// RUN: %clang_cc1 -DCK28 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY27 %s
// SIMD-ONLY27-NOT: {{__kmpc|__tgt}}
#ifdef CK28

// CK28-LABEL: @.__omp_offloading_{{.*}}explicit_maps_pointer_references{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK28: [[SIZE00:@.+]] = private {{.*}}constant [1 x i64] [i64 {{8|4}}]
// CK28: [[MTYPE00:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK28-LABEL: @.__omp_offloading_{{.*}}explicit_maps_pointer_references{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK28: [[SIZE01:@.+]] = private {{.*}}constant [2 x i64] [i64 400, i64 {{4|8}}]
// CK28: [[MTYPE01:@.+]] = private {{.*}}constant [2 x i64] [i64 35, i64 16384]

// CK28-LABEL: explicit_maps_pointer_references{{.*}}(
void explicit_maps_pointer_references (int *p){
  int *&a = p;

// Region 00
// CK28-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK28-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK28-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK28-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK28-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CK28-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK28-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK28-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK28-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK28-DAG: store ptr [[VAR0:%.+]], ptr [[BP0]]
// CK28-DAG: store ptr [[VAR1:%.+]], ptr [[P0]]
// CK28-DAG: [[VAR0]] = load ptr, ptr [[VAR00:%.+]],
// CK28-DAG: [[VAR1]] = load ptr, ptr [[VAR11:%.+]],

// CK28: call void [[CALL00:@.+]](ptr {{[^,]+}})
#pragma omp target map(a)
  {
    ++a;
  }

// Region 01

//  &(ref_ptee(a)[0]), &(ref_ptee(a)[2]), 100 * sizeof(int), TO | FROM | PARAM
//  &(ref_ptee(a)), &(ref_ptee(a)[2]), sizeof(ref_ptee(a)), ATTACH

// CK28-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK28-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK28-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK28-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK28-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CK28-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK28-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK28-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK28-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK28-DAG: store ptr [[RRA:%.+]], ptr [[BP0]]
// CK28-DAG: store ptr [[RRA2:%.+]], ptr [[P0]]
// CK28-DAG: [[RRA]] = load ptr, ptr [[RA:%.+]],
// CK28-DAG: [[RA]] = load ptr, ptr [[A:%.+]],
// CK28-DAG: [[RRA2]] = getelementptr inbounds nuw i32, ptr [[RRA_1:%.+]], i{{64|32}} 2
// CK28-DAG: [[RRA_1]] = load ptr, ptr [[RA_1:%.+]],
// CK28-DAG: [[RA_1]] = load ptr, ptr [[A]]

// CK28-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
// CK28-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
// CK28-DAG: store ptr [[RA_2:%.+]], ptr [[BP1]]
// CK28-DAG: store ptr [[RRA2_2:%.+]], ptr [[P1]]
// CK28-DAG: [[RA_2]] = load ptr, ptr [[A]]
// CK28-DAG: [[RRA2_2]] = getelementptr inbounds nuw i32, ptr [[RRA_2:%.+]], i{{64|32}} 2
// CK28-DAG: [[RRA_2]] = load ptr, ptr [[RA_3:%.+]],
// CK28-DAG: [[RA_3]] = load ptr, ptr [[A]]

// CK28: call void [[CALL01:@.+]](ptr {{[^,]+}})
#pragma omp target map(a [2:100])
  {
    ++a;
  }
}
#endif // CK28
#endif
