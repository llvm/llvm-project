// expected-no-diagnostics
#ifndef HEADER
#define HEADER

///==========================================================================///
// RUN: %clang_cc1 -DCK12 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK12 --check-prefix CK12-64
// RUN: %clang_cc1 -DCK12 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK12 --check-prefix CK12-64
// RUN: %clang_cc1 -DCK12 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK12 --check-prefix CK12-32
// RUN: %clang_cc1 -DCK12 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK12 --check-prefix CK12-32

// RUN: %clang_cc1 -DCK12 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK12 --check-prefix CK12-64
// RUN: %clang_cc1 -DCK12 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK12 --check-prefix CK12-64
// RUN: %clang_cc1 -DCK12 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK12 --check-prefix CK12-32
// RUN: %clang_cc1 -DCK12 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK12 --check-prefix CK12-32

// RUN: %clang_cc1 -DCK12 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK12 --check-prefix CK12-64
// RUN: %clang_cc1 -DCK12 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK12 --check-prefix CK12-64
// RUN: %clang_cc1 -DCK12 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK12 --check-prefix CK12-32
// RUN: %clang_cc1 -DCK12 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK12 --check-prefix CK12-32

// RUN: %clang_cc1 -DCK12 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY11 %s
// RUN: %clang_cc1 -DCK12 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY11 %s
// RUN: %clang_cc1 -DCK12 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY11 %s
// RUN: %clang_cc1 -DCK12 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY11 %s
// SIMD-ONLY11-NOT: {{__kmpc|__tgt}}
#ifdef CK12

// For a 32-bit targets, the value doesn't fit the size of the pointer,
// therefore it is passed by reference with a map 'to' specification.

// CK12-LABEL: @.__omp_offloading_{{.*}}implicit_maps_float_complex{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0

// CK12-DAG: [[SIZES:@.+]] = {{.+}}constant [1 x i64] [i64 8]
// Map types: OMP_MAP_PRIVATE_VAL + OMP_MAP_TARGET_PARAM | OMP_MAP_IMPLICIT = 800
// CK12-64-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i64] [i64 800]
// Map types: OMP_MAP_TO  | OMP_MAP_PRIVATE | OMP_MAP_TARGET_PARAM | OMP_MAP_IMPLICIT = 673
// CK12-32-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i64] [i64 673]

// CK12-LABEL: implicit_maps_float_complex{{.*}}(
void implicit_maps_float_complex (int a){
  float _Complex fc = (float)a;

// CK12-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK12-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK12-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK12-DAG: [[PGEP:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK12-DAG: store ptr [[PGEP:%.+]], ptr [[BPARG]]
// CK12-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
// CK12-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
// CK12-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
// CK12-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0

// CK12-64-DAG: store i[[sz:64|32]] [[VAL:%[^,]+]], ptr [[BP1]]
// CK12-64-DAG: store i[[sz]] [[VAL]], ptr [[P1]]
// CK12-64-DAG: [[VAL]] = load i[[sz]], ptr [[ADDR:%.+]],
// CK12-64-DAG: store { float, float } {{.+}}, ptr [[ADDR]],

// CK12-32-DAG: store ptr [[DECL:%[^,]+]], ptr [[BP1]]
// CK12-32-DAG: store ptr [[DECL]], ptr [[P1]]

// CK12-64: call void [[KERNEL:@.+]](i[[sz]] [[VAL]])
// CK12-32: call void [[KERNEL:@.+]](ptr [[DECL]])
#pragma omp target
  {
    fc *= fc;
  }
}

// CK12-64: define internal void [[KERNEL]](i[[sz]] noundef [[ARG:%.+]])
// CK12-64: [[ADDR:%.+]] = alloca i[[sz]],
// CK12-64: store i[[sz]] [[ARG]], ptr [[ADDR]],
// CK12-64: {{.+}} = getelementptr inbounds nuw { float, float }, ptr [[ADDR]], i32 0, i32 0

// CK12-32: define internal void [[KERNEL]](ptr {{.+}}[[ARG:%.+]])
// CK12-32: [[ADDR:%.+]] = alloca ptr,
// CK12-32: store ptr [[ARG]], ptr [[ADDR]],
// CK12-32: [[REF:%.+]] = load ptr, ptr [[ADDR]],
// CK12-32: {{.+}} = getelementptr inbounds nuw { float, float }, ptr [[REF]], i32 0, i32 0
#endif // CK12
#endif
