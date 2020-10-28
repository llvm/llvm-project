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

  // CK12-DAG: call i32 @__tgt_target_mapper(i64 {{.+}}, i8* {{.+}}, i32 1, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZES]]{{.+}}, {{.+}}[[TYPES]]{{.+}}, i8** null)
  // CK12-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK12-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK12-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK12-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0

  // CK12-64-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to i[[sz:64|32]]*
  // CK12-64-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i[[sz]]*
  // CK12-64-DAG: store i[[sz]] [[VAL:%[^,]+]], i[[sz]]* [[CBP1]]
  // CK12-64-DAG: store i[[sz]] [[VAL]], i[[sz]]* [[CP1]]
  // CK12-64-DAG: [[VAL]] = load i[[sz]], i[[sz]]* [[ADDR:%.+]],
  // CK12-64-DAG: [[CADDR:%.+]] = bitcast i[[sz]]* [[ADDR]] to { float, float }*
  // CK12-64-DAG: store { float, float } {{.+}}, { float, float }* [[CADDR]],

  // CK12-32-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to { float, float }**
  // CK12-32-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to { float, float }**
  // CK12-32-DAG: store { float, float }* [[DECL:%[^,]+]], { float, float }** [[CBP1]]
  // CK12-32-DAG: store { float, float }* [[DECL]], { float, float }** [[CP1]]

  // CK12-64: call void [[KERNEL:@.+]](i[[sz]] [[VAL]])
  // CK12-32: call void [[KERNEL:@.+]]({ float, float }* [[DECL]])
  #pragma omp target
  {
    fc *= fc;
  }
}

// CK12-64: define internal void [[KERNEL]](i[[sz]] [[ARG:%.+]])
// CK12-64: [[ADDR:%.+]] = alloca i[[sz]],
// CK12-64: store i[[sz]] [[ARG]], i[[sz]]* [[ADDR]],
// CK12-64: [[CADDR:%.+]] = bitcast i[[sz]]* [[ADDR]] to { float, float }*
// CK12-64: {{.+}} = getelementptr inbounds { float, float }, { float, float }* [[CADDR]], i32 0, i32 0

// CK12-32: define internal void [[KERNEL]]({ float, float }* {{.+}}[[ARG:%.+]])
// CK12-32: [[ADDR:%.+]] = alloca { float, float }*,
// CK12-32: store { float, float }* [[ARG]], { float, float }** [[ADDR]],
// CK12-32: [[REF:%.+]] = load { float, float }*, { float, float }** [[ADDR]],
// CK12-32: {{.+}} = getelementptr inbounds { float, float }, { float, float }* [[REF]], i32 0, i32 0
#endif // CK12
#endif
