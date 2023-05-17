// expected-no-diagnostics
#ifndef HEADER
#define HEADER

///==========================================================================///
// RUN: %clang_cc1 -DCK13 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK13
// RUN: %clang_cc1 -DCK13 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK13
// RUN: %clang_cc1 -DCK13 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK13
// RUN: %clang_cc1 -DCK13 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK13

// RUN: %clang_cc1 -DCK13 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK13
// RUN: %clang_cc1 -DCK13 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK13
// RUN: %clang_cc1 -DCK13 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK13
// RUN: %clang_cc1 -DCK13 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK13

// RUN: %clang_cc1 -DCK13 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK13
// RUN: %clang_cc1 -DCK13 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK13
// RUN: %clang_cc1 -DCK13 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK13
// RUN: %clang_cc1 -DCK13 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK13

// RUN: %clang_cc1 -DCK13 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY12 %s
// RUN: %clang_cc1 -DCK13 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY12 %s
// RUN: %clang_cc1 -DCK13 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY12 %s
// RUN: %clang_cc1 -DCK13 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY12 %s
// SIMD-ONLY12-NOT: {{__kmpc|__tgt}}
#ifdef CK13

// CK13-LABEL: @.__omp_offloading_{{.*}}implicit_maps_variable_length_array{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0

// CK13-DAG: [[SIZES:@.+]] = {{.+}}constant [3 x i64] [i64 {{8|4}}, i64 {{8|4}}, i64 0]
// Map types:
//  - OMP_MAP_PRIVATE_VAL + OMP_MAP_TARGET_PARAM + OMP_MAP_IMPLICIT = 800 (vla size)
//  - OMP_MAP_PRIVATE_VAL + OMP_MAP_TARGET_PARAM + OMP_MAP_IMPLICIT = 800 (vla size)
//  - OMP_MAP_TO + OMP_MAP_FROM + OMP_MAP_TARGET_PARAM | OMP_MAP_IMPLICIT = 547
// CK13-DAG: [[TYPES:@.+]] = {{.+}}constant [3 x i64] [i64 800, i64 800, i64 547]

// CK13-LABEL: implicit_maps_variable_length_array{{.*}}(
void implicit_maps_variable_length_array (int a){
  double vla[2][a];

// CK13-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK13-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK13-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK13-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK13-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CK13-DAG: [[SARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 4
// CK13-DAG: store ptr [[SGEP:%.+]], ptr [[SARG]]
// CK13-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
// CK13-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
// CK13-DAG: [[SGEP]] = getelementptr inbounds {{.+}}[[SS:%[^,]+]], i32 0, i32 0

// CK13-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
// CK13-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
// CK13-DAG: store i[[sz:64|32]] 2, ptr [[BP0]]
// CK13-DAG: store i[[sz]] 2, ptr [[P0]]

// CK13-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 1
// CK13-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 1
// CK13-DAG: store i[[sz]] [[VAL:%.+]], ptr [[BP1]]
// CK13-DAG: store i[[sz]] [[VAL]], ptr [[P1]]

// CK13-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 2
// CK13-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 2
// CK13-DAG: [[S2:%.+]] = getelementptr inbounds {{.+}}[[SS]], i32 0, i32 2
// CK13-DAG: store ptr [[DECL:%.+]], ptr [[BP2]]
// CK13-DAG: store ptr [[DECL]], ptr [[P2]]
// CK13-DAG: store i64 [[VALS2:%.+]], ptr [[S2]],
// CK13-DAG: [[VALS2]] = {{mul nuw i64 %.+, 8|sext i32 %.+ to i64}}

// CK13: call void [[KERNEL:@.+]](i[[sz]] {{.+}}, i[[sz]] {{.+}}, ptr [[DECL]])
#pragma omp target
  {
    vla[1][3] += 1.0;
  }
}

// CK13: define internal void [[KERNEL]](i[[sz]] noundef [[VLA0:%.+]], i[[sz]] noundef [[VLA1:%.+]], ptr {{.*}}[[ARG:%.+]])
// CK13: [[ADDR0:%.+]] = alloca i[[sz]],
// CK13: [[ADDR1:%.+]] = alloca i[[sz]],
// CK13: [[ADDR2:%.+]] = alloca ptr,
// CK13: store i[[sz]] [[VLA0]], ptr [[ADDR0]],
// CK13: store i[[sz]] [[VLA1]], ptr [[ADDR1]],
// CK13: store ptr [[ARG]], ptr [[ADDR2]],
// CK13: {{.+}} = load i[[sz]],  ptr [[ADDR0]],
// CK13: {{.+}} = load i[[sz]],  ptr [[ADDR1]],
// CK13: [[REF:%.+]] = load ptr, ptr [[ADDR2]],
// CK13: {{.+}} = getelementptr inbounds double, ptr [[REF]], i[[sz]] %{{.+}}
#endif // CK13
#endif
