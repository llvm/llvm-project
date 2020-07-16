// expected-no-diagnostics
#ifndef HEADER
#define HEADER

#ifdef CK1

///==========================================================================///
// RUN: %clang_cc1 -DCK1 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK1
// RUN: %clang_cc1 -DCK1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK1
// RUN: %clang_cc1 -DCK1 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK1
// RUN: %clang_cc1 -DCK1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK1

// RUN: %clang_cc1 -DCK1 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY10 %s
// RUN: %clang_cc1 -DCK1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY10 %s
// RUN: %clang_cc1 -DCK1 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY10 %s
// RUN: %clang_cc1 -DCK1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY10 %s
// SIMD-ONLY10-NOT: {{__kmpc|__tgt}}

// CK1-LABEL: @.__omp_offloading_{{.*}}implicit_maps_double_complex{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0

// CK1-DAG: [[SIZES:@.+]] = {{.+}}constant [1 x i64] [i64 16]
// Map types: OMP_MAP_TARGET_PARAM | OMP_MAP_IMPLICIT = 544
// CK1-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i64] [i64 544]

// CK1-LABEL: implicit_maps_double_complex{{.*}}(
void implicit_maps_double_complex (int a){
  double _Complex dc = (double)a;

  // CK1-DAG: call i32 @__tgt_target_mapper(i64 {{.+}}, i8* {{.+}}, i32 1, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZES]]{{.+}}, {{.+}}[[TYPES]]{{.+}}, i8** null)
  // CK1-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK1-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK1-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK1-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK1-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to { double, double }**
  // CK1-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to { double, double }**
  // CK1-DAG: store { double, double }* [[PTR:%[^,]+]], { double, double }** [[CBP1]]
  // CK1-DAG: store { double, double }* [[PTR]], { double, double }** [[CP1]]

  // CK1: call void [[KERNEL:@.+]]({ double, double }* [[PTR]])
  #pragma omp target defaultmap(alloc:scalar)
  {
   dc *= dc;
  }
}

// CK1: define internal void [[KERNEL]]({ double, double }* {{.*}}[[ARG:%.+]])
// CK1: [[ADDR:%.+]] = alloca { double, double }*,
// CK1: store { double, double }* [[ARG]], { double, double }** [[ADDR]],
// CK1: [[REF:%.+]] = load { double, double }*, { double, double }** [[ADDR]],
// CK1: {{.+}} = getelementptr inbounds { double, double }, { double, double }* [[REF]], i32 0, i32 0
#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK2 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK2
// RUN: %clang_cc1 -DCK2 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK2
// RUN: %clang_cc1 -DCK2 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK2
// RUN: %clang_cc1 -DCK2 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK2

// RUN: %clang_cc1 -DCK2 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY10 %s
// RUN: %clang_cc1 -DCK2 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY10 %s
// RUN: %clang_cc1 -DCK2 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY10 %s
// RUN: %clang_cc1 -DCK2 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY10 %s
// SIMD-ONLY10-NOT: {{__kmpc|__tgt}}
#ifdef CK2

// CK2-LABEL: @.__omp_offloading_{{.*}}implicit_maps_double_complex{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0

// CK2-DAG: [[SIZES:@.+]] = {{.+}}constant [1 x i64] [i64 16]
// Map types: OMP_MAP_TO  | OMP_MAP_TARGET_PARAM | OMP_MAP_IMPLICIT = 545
// CK2-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i64] [i64 545]

// CK2-LABEL: implicit_maps_double_complex{{.*}}(
void implicit_maps_double_complex (int a){
  double _Complex dc = (double)a;

  // CK2-DAG: call i32 @__tgt_target_mapper(i64 {{.+}}, i8* {{.+}}, i32 1, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZES]]{{.+}}, {{.+}}[[TYPES]]{{.+}}, i8** null)
  // CK2-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK2-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK2-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK2-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK2-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to { double, double }**
  // CK2-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to { double, double }**
  // CK2-DAG: store { double, double }* [[PTR:%[^,]+]], { double, double }** [[CBP1]]
  // CK2-DAG: store { double, double }* [[PTR]], { double, double }** [[CP1]]

  // CK2: call void [[KERNEL:@.+]]({ double, double }* [[PTR]])
  #pragma omp target defaultmap(to:scalar)
  {
   dc *= dc;
  }
}

// CK2: define internal void [[KERNEL]]({ double, double }* {{.*}}[[ARG:%.+]])
// CK2: [[ADDR:%.+]] = alloca { double, double }*,
// CK2: store { double, double }* [[ARG]], { double, double }** [[ADDR]],
// CK2: [[REF:%.+]] = load { double, double }*, { double, double }** [[ADDR]],
// CK2: {{.+}} = getelementptr inbounds { double, double }, { double, double }* [[REF]], i32 0, i32 0
#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK3 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK3
// RUN: %clang_cc1 -DCK3 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK3
// RUN: %clang_cc1 -DCK3 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK3
// RUN: %clang_cc1 -DCK3 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK3

// RUN: %clang_cc1 -DCK3 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY10 %s
// RUN: %clang_cc1 -DCK3 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY10 %s
// RUN: %clang_cc1 -DCK3 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY10 %s
// RUN: %clang_cc1 -DCK3 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY10 %s
// SIMD-ONLY10-NOT: {{__kmpc|__tgt}}
#ifdef CK3

// CK3-LABEL: @.__omp_offloading_{{.*}}implicit_maps_double_complex{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0

// CK3-DAG: [[SIZES:@.+]] = {{.+}}constant [1 x i64] [i64 16]
// Map types: OMP_MAP_FROM | OMP_MAP_TARGET_PARAM | OMP_MAP_IMPLICIT = 546
// CK3-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i64] [i64 546]

// CK3-LABEL: implicit_maps_double_complex{{.*}}(
void implicit_maps_double_complex (int a){
  double _Complex dc = (double)a;

  // CK3-DAG: call i32 @__tgt_target_mapper(i64 {{.+}}, i8* {{.+}}, i32 1, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZES]]{{.+}}, {{.+}}[[TYPES]]{{.+}}, i8** null)
  // CK3-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK3-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK3-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK3-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK3-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to { double, double }**
  // CK3-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to { double, double }**
  // CK3-DAG: store { double, double }* [[PTR:%[^,]+]], { double, double }** [[CBP1]]
  // CK3-DAG: store { double, double }* [[PTR]], { double, double }** [[CP1]]

  // CK3: call void [[KERNEL:@.+]]({ double, double }* [[PTR]])
  #pragma omp target defaultmap(from:scalar)
  {
   dc *= dc;
  }
}

// CK3: define internal void [[KERNEL]]({ double, double }* {{.*}}[[ARG:%.+]])
// CK3: [[ADDR:%.+]] = alloca { double, double }*,
// CK3: store { double, double }* [[ARG]], { double, double }** [[ADDR]],
// CK3: [[REF:%.+]] = load { double, double }*, { double, double }** [[ADDR]],
// CK3: {{.+}} = getelementptr inbounds { double, double }, { double, double }* [[REF]], i32 0, i32 0
#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK4 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK4 --check-prefix CK4-64
// RUN: %clang_cc1 -DCK4 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK4  --check-prefix CK4-64
// RUN: %clang_cc1 -DCK4 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK4  --check-prefix CK4-32
// RUN: %clang_cc1 -DCK4 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK4  --check-prefix CK4-32

// RUN: %clang_cc1 -DCK4 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY6 %s
// RUN: %clang_cc1 -DCK4 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY6 %s
// RUN: %clang_cc1 -DCK4 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY6 %s
// RUN: %clang_cc1 -DCK4 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY6 %s
// SIMD-ONLY6-NOT: {{__kmpc|__tgt}}
#ifdef CK4

// For a 32-bit targets, the value doesn't fit the size of the pointer,
// therefore it is passed by reference with a map 'to' specification.

// CK4-LABEL: @.__omp_offloading_{{.*}}implicit_maps_double{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0

// CK4-DAG: [[SIZES:@.+]] = {{.+}}constant [1 x i64] [i64 8]
// Map types: OMP_MAP_PRIVATE_VAL | OMP_MAP_TARGET_PARAM | OMP_MAP_IMPLICIT = 800
// CK4-64-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i64] [i64 800]
// Map types: OMP_MAP_TO  | OMP_MAP_PRIVATE | OMP_MAP_TARGET_PARAM | OMP_MAP_IMPLICIT = 673
// CK4-32-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i64] [i64 673]

// CK4-LABEL: implicit_maps_double{{.*}}(
void implicit_maps_double (int a){
  double d = (double)a;

  // CK4-DAG: call i32 @__tgt_target_mapper(i64 {{.+}}, i8* {{.+}}, i32 1, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZES]]{{.+}}, {{.+}}[[TYPES]]{{.+}}, i8** null)
  // CK4-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK4-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK4-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK4-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0

  // CK4-64-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to i[[sz:64|32]]*
  // CK4-64-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i[[sz]]*
  // CK4-64-DAG: store i[[sz]] [[VAL:%[^,]+]], i[[sz]]* [[CBP1]]
  // CK4-64-DAG: store i[[sz]] [[VAL]], i[[sz]]* [[CP1]]
  // CK4-64-DAG: [[VAL]] = load i[[sz]], i[[sz]]* [[ADDR:%.+]],
  // CK4-64-64-DAG: [[CADDR:%.+]] = bitcast i[[sz]]* [[ADDR]] to double*
  // CK4-64-64-DAG: store double {{.+}}, double* [[CADDR]],

  // CK4-32-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to double**
  // CK4-32-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to double**
  // CK4-32-DAG: store double* [[DECL:%[^,]+]], double** [[CBP1]]
  // CK4-32-DAG: store double* [[DECL]], double** [[CP1]]

  // CK4-64: call void [[KERNEL:@.+]](i[[sz]] [[VAL]])
  // CK4-32: call void [[KERNEL:@.+]](double* [[DECL]])
  #pragma omp target defaultmap(firstprivate:scalar)
  {
    d += 1.0;
  }
}

// CK4-64: define internal void [[KERNEL]](i[[sz]] [[ARG:%.+]])
// CK4-64: [[ADDR:%.+]] = alloca i[[sz]],
// CK4-64: store i[[sz]] [[ARG]], i[[sz]]* [[ADDR]],
// CK4-64: [[CADDR:%.+]] = bitcast i64* [[ADDR]] to double*
// CK4-64: {{.+}} = load double, double* [[CADDR]],

// CK4-32: define internal void [[KERNEL]](double* {{.+}}[[ARG:%.+]])
// CK4-32: [[ADDR:%.+]] = alloca double*,
// CK4-32: store double* [[ARG]], double** [[ADDR]],
// CK4-32: [[REF:%.+]] = load double*, double** [[ADDR]],
// CK4-32: {{.+}} = load double, double* [[REF]],

#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK5 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK5
// RUN: %clang_cc1 -DCK5 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK5
// RUN: %clang_cc1 -DCK5 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK5
// RUN: %clang_cc1 -DCK5 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK5

// RUN: %clang_cc1 -DCK5 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY8 %s
// RUN: %clang_cc1 -DCK5 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY8 %s
// RUN: %clang_cc1 -DCK5 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY8 %s
// RUN: %clang_cc1 -DCK5 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY8 %s
// SIMD-ONLY8-NOT: {{__kmpc|__tgt}}
#ifdef CK5

// CK5-LABEL: @.__omp_offloading_{{.*}}implicit_maps_array{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0

// CK5-DAG: [[SIZES:@.+]] = {{.+}}constant [1 x i64] [i64 16]
// Map types: OMP_MAP_TARGET_PARAM | OMP_MAP_IMPLICIT = 544
// CK5-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i64] [i64 544]

// CK5-LABEL: implicit_maps_array{{.*}}(
void implicit_maps_array (int a){
  double darr[2] = {(double)a, (double)a};

  // CK5-DAG: call i32 @__tgt_target_mapper(i64 {{.+}}, i8* {{.+}}, i32 1, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZES]]{{.+}}, {{.+}}[[TYPES]]{{.+}}, i8** null)
  // CK5-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK5-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK5-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK5-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK5-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to [2 x double]**
  // CK5-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to [2 x double]**
  // CK5-DAG: store [2 x double]* [[DECL:%[^,]+]], [2 x double]** [[CBP1]]
  // CK5-DAG: store [2 x double]* [[DECL]], [2 x double]** [[CP1]]

  // CK5: call void [[KERNEL:@.+]]([2 x double]* [[DECL]])
  #pragma omp target defaultmap(alloc: aggregate)
  {
    darr[0] += 1.0;
    darr[1] += 1.0;
  }
}

// CK5: define internal void [[KERNEL]]([2 x double]* {{.+}}[[ARG:%.+]])
// CK5: [[ADDR:%.+]] = alloca [2 x double]*,
// CK5: store [2 x double]* [[ARG]], [2 x double]** [[ADDR]],
// CK5: [[REF:%.+]] = load [2 x double]*, [2 x double]** [[ADDR]],
// CK5: {{.+}} = getelementptr inbounds [2 x double], [2 x double]* [[REF]], i{{64|32}} 0, i{{64|32}} 0
#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK6 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK6
// RUN: %clang_cc1 -DCK6 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK6
// RUN: %clang_cc1 -DCK6 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK6
// RUN: %clang_cc1 -DCK6 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK6

// RUN: %clang_cc1 -DCK6 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY8 %s
// RUN: %clang_cc1 -DCK6 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY8 %s
// RUN: %clang_cc1 -DCK6 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY8 %s
// RUN: %clang_cc1 -DCK6 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY8 %s
// SIMD-ONLY8-NOT: {{__kmpc|__tgt}}
#ifdef CK6

// CK6-LABEL: @.__omp_offloading_{{.*}}implicit_maps_array{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0

// CK6-DAG: [[SIZES:@.+]] = {{.+}}constant [1 x i64] [i64 16]
// Map types: OMP_MAP_TO | OMP_MAP_TARGET_PARAM | OMP_MAP_IMPLICIT = 545
// CK6-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i64] [i64 545]

// CK6-LABEL: implicit_maps_array{{.*}}(
void implicit_maps_array (int a){
  double darr[2] = {(double)a, (double)a};

  // CK6-DAG: call i32 @__tgt_target_mapper(i64 {{.+}}, i8* {{.+}}, i32 1, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZES]]{{.+}}, {{.+}}[[TYPES]]{{.+}}, i8** null)
  // CK6-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK6-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK6-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK6-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK6-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to [2 x double]**
  // CK6-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to [2 x double]**
  // CK6-DAG: store [2 x double]* [[DECL:%[^,]+]], [2 x double]** [[CBP1]]
  // CK6-DAG: store [2 x double]* [[DECL]], [2 x double]** [[CP1]]

  // CK6: call void [[KERNEL:@.+]]([2 x double]* [[DECL]])
#pragma omp target defaultmap(to)
  {
    darr[0] += 1.0;
    darr[1] += 1.0;
  }
}

// CK6: define internal void [[KERNEL]]([2 x double]* {{.+}}[[ARG:%.+]])
// CK6: [[ADDR:%.+]] = alloca [2 x double]*,
// CK6: store [2 x double]* [[ARG]], [2 x double]** [[ADDR]],
// CK6: [[REF:%.+]] = load [2 x double]*, [2 x double]** [[ADDR]],
// CK6: {{.+}} = getelementptr inbounds [2 x double], [2 x double]* [[REF]], i{{64|32}} 0, i{{64|32}} 0
#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK7 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK7
// RUN: %clang_cc1 -DCK7 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK7
// RUN: %clang_cc1 -DCK7 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK7
// RUN: %clang_cc1 -DCK7 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK7

// RUN: %clang_cc1 -DCK7 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY8 %s
// RUN: %clang_cc1 -DCK7 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY8 %s
// RUN: %clang_cc1 -DCK7 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY8 %s
// RUN: %clang_cc1 -DCK7 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY8 %s
// SIMD-ONLY8-NOT: {{__kmpc|__tgt}}
#ifdef CK7

// CK7-LABEL: @.__omp_offloading_{{.*}}implicit_maps_array{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0

// CK7-DAG: [[SIZES:@.+]] = {{.+}}constant [1 x i64] [i64 16]
// Map types: OMP_MAP_FROM | OMP_MAP_TARGET_PARAM | OMP_MAP_IMPLICIT = 546
// CK7-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i64] [i64 546]

// CK7-LABEL: implicit_maps_array{{.*}}(
void implicit_maps_array (int a){
  double darr[2] = {(double)a, (double)a};

  // CK7-DAG: call i32 @__tgt_target_mapper(i64 {{.+}}, i8* {{.+}}, i32 1, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZES]]{{.+}}, {{.+}}[[TYPES]]{{.+}}, i8** null)
  // CK7-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK7-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK7-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK7-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK7-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to [2 x double]**
  // CK7-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to [2 x double]**
  // CK7-DAG: store [2 x double]* [[DECL:%[^,]+]], [2 x double]** [[CBP1]]
  // CK7-DAG: store [2 x double]* [[DECL]], [2 x double]** [[CP1]]

  // CK7: call void [[KERNEL:@.+]]([2 x double]* [[DECL]])
  #pragma omp target defaultmap(from: aggregate)
  {
    darr[0] += 1.0;
    darr[1] += 1.0;
  }
}

// CK7: define internal void [[KERNEL]]([2 x double]* {{.+}}[[ARG:%.+]])
// CK7: [[ADDR:%.+]] = alloca [2 x double]*,
// CK7: store [2 x double]* [[ARG]], [2 x double]** [[ADDR]],
// CK7: [[REF:%.+]] = load [2 x double]*, [2 x double]** [[ADDR]],
// CK7: {{.+}} = getelementptr inbounds [2 x double], [2 x double]* [[REF]], i{{64|32}} 0, i{{64|32}} 0
#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK8 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK8
// RUN: %clang_cc1 -DCK8 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK8
// RUN: %clang_cc1 -DCK8 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK8
// RUN: %clang_cc1 -DCK8 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK8

// RUN: %clang_cc1 -DCK8 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY8 %s
// RUN: %clang_cc1 -DCK8 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY8 %s
// RUN: %clang_cc1 -DCK8 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY8 %s
// RUN: %clang_cc1 -DCK8 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY8 %s
// SIMD-ONLY8-NOT: {{__kmpc|__tgt}}
#ifdef CK8

// CK8-LABEL: @.__omp_offloading_{{.*}}implicit_maps_array{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0

// CK8-DAG: [[SIZES:@.+]] = {{.+}}constant [1 x i64] [i64 16]
// Map types: OMP_MAP_TO | OMP_MAP_FROM | OMP_MAP_TARGET_PARAM | OMP_MAP_IMPLICIT = 547
// CK8-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i64] [i64 547]

// CK8-LABEL: implicit_maps_array{{.*}}(
void implicit_maps_array (int a){
  double darr[2] = {(double)a, (double)a};

  // CK8-DAG: call i32 @__tgt_target_mapper(i64 {{.+}}, i8* {{.+}}, i32 1, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZES]]{{.+}}, {{.+}}[[TYPES]]{{.+}}, i8** null)
  // CK8-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK8-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK8-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK8-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK8-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to [2 x double]**
  // CK8-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to [2 x double]**
  // CK8-DAG: store [2 x double]* [[DECL:%[^,]+]], [2 x double]** [[CBP1]]
  // CK8-DAG: store [2 x double]* [[DECL]], [2 x double]** [[CP1]]

  // CK8: call void [[KERNEL:@.+]]([2 x double]* [[DECL]])
#pragma omp target defaultmap(tofrom)
  {
    darr[0] += 1.0;
    darr[1] += 1.0;
  }
}

// CK8: define internal void [[KERNEL]]([2 x double]* {{.+}}[[ARG:%.+]])
// CK8: [[ADDR:%.+]] = alloca [2 x double]*,
// CK8: store [2 x double]* [[ARG]], [2 x double]** [[ADDR]],
// CK8: [[REF:%.+]] = load [2 x double]*, [2 x double]** [[ADDR]],
// CK8: {{.+}} = getelementptr inbounds [2 x double], [2 x double]* [[REF]], i{{64|32}} 0, i{{64|32}} 0
#endif

///==========================================================================///
// RUN: %clang_cc1 -DCK9 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK9 --check-prefix CK9-64
// RUN: %clang_cc1 -DCK9 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK9 --check-prefix CK9-64
// RUN: %clang_cc1 -DCK9 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK9 --check-prefix CK9-32
// RUN: %clang_cc1 -DCK9 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK9 --check-prefix CK9-32

// RUN: %clang_cc1 -DCK9 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY26 %s
// RUN: %clang_cc1 -DCK9 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY26 %s
// RUN: %clang_cc1 -DCK9 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY26 %s
// RUN: %clang_cc1 -DCK9 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY26 %s
// SIMD-ONLY26-NOT: {{__kmpc|__tgt}}

#ifdef CK9


// CK9-LABEL: @.__omp_offloading_{{.*}}zero_size_section_and_private_maps{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK9: [[SIZE09:@.+]] = private {{.*}}constant [1 x i64] [i64 40]
// CK9: [[MTYPE09:@.+]] = private {{.*}}constant [1 x i64] [i64 673]


// CK9-LABEL: zero_size_section_and_private_maps{{.*}}(
void zero_size_section_and_private_maps (int ii){
  int pvtArr[10];

  // Region 09
  // CK9-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE09]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE09]]{{.+}}, i8** null)
  // CK9-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK9-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK9-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK9-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK9-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [10 x i32]**
  // CK9-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to [10 x i32]**
  // CK9-DAG: store [10 x i32]* [[VAR0:%.+]], [10 x i32]** [[CBP0]]
  // CK9-DAG: store [10 x i32]* [[VAR0]], [10 x i32]** [[CP0]]

  // CK9: call void [[CALL09:@.+]]([10 x i32]* {{[^,]+}})
  #pragma omp target defaultmap(firstprivate:aggregate)
  {
    pvtArr[5]++;
  }

}


#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK10 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK10
// RUN: %clang_cc1 -DCK10 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK10
// RUN: %clang_cc1 -DCK10 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK10
// RUN: %clang_cc1 -DCK10 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK10

// RUN: %clang_cc1 -DCK10 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY8 %s
// RUN: %clang_cc1 -DCK10 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY8 %s
// RUN: %clang_cc1 -DCK10 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY8 %s
// RUN: %clang_cc1 -DCK10 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY8 %s
// SIMD-ONLY8-NOT: {{__kmpc|__tgt}}
#ifdef CK10


// CK10-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0

// CK10: [[SIZE:@.+]] = private {{.*}}constant [1 x i64] [i64 {{8|4}}]
// Map types: OMP_MAP_TARGET_PARAM | OMP_MAP_IMPLICIT = 544
// CK10: [[MTYPE:@.+]] = private {{.*}}constant [1 x i64] [i64 544]

// CK10-LABEL: explicit_maps_single{{.*}}(
void explicit_maps_single (){
  int *pa;

  // CK10-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE]]{{.+}}, i8** null)
  // CK10-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK10-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK10-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK10-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK10-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i32***
  // CK10-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32***
  // CK10-DAG: store i32** [[VAR0:%.+]], i32*** [[CBP0]]
  // CK10-DAG: store i32** [[VAR0]], i32*** [[CP0]]

  // CK10: call void [[CALL:@.+]](i32** {{[^,]+}})
  #pragma omp target defaultmap(alloc: pointer)
  {
    pa[50]++;
  }
}

  // CK10: define {{.+}}[[CALL]]
#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK11 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK11
// RUN: %clang_cc1 -DCK11 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK11
// RUN: %clang_cc1 -DCK11 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK11
// RUN: %clang_cc1 -DCK11 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK11

// RUN: %clang_cc1 -DCK11 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY8 %s
// RUN: %clang_cc1 -DCK11 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY8 %s
// RUN: %clang_cc1 -DCK11 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY8 %s
// RUN: %clang_cc1 -DCK11 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY8 %s
// SIMD-ONLY8-NOT: {{__kmpc|__tgt}}
#ifdef CK11


// CK11-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0

// CK11: [[SIZE09:@.+]] = private {{.*}}constant [1 x i64] [i64 {{8|4}}]
// Map types: OMP_MAP_TO | OMP_MAP_TARGET_PARAM | OMP_MAP_IMPLICIT = 545
// CK11: [[MTYPE09:@.+]] = private {{.*}}constant [1 x i64] [i64 545]

// CK11-LABEL: explicit_maps_single{{.*}}(
void explicit_maps_single (){
  int *pa;

  // CK11-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE09]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE09]]{{.+}}, i8** null)
  // CK11-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK11-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK11-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK11-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK11-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i32***
  // CK11-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32***
  // CK11-DAG: store i32** [[VAR0:%.+]], i32*** [[CBP0]]
  // CK11-DAG: store i32** [[VAR0]], i32*** [[CP0]]

  // CK11: call void [[CALL09:@.+]](i32** {{[^,]+}})
  #pragma omp target defaultmap(to: pointer)
  {
    pa[50]++;
  }
}

  // CK11: define {{.+}}[[CALL09]]
#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK12 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK12
// RUN: %clang_cc1 -DCK12 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK12
// RUN: %clang_cc1 -DCK12 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK12
// RUN: %clang_cc1 -DCK12 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK12

// RUN: %clang_cc1 -DCK12 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY8 %s
// RUN: %clang_cc1 -DCK12 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY8 %s
// RUN: %clang_cc1 -DCK12 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY8 %s
// RUN: %clang_cc1 -DCK12 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY8 %s
// SIMD-ONLY8-NOT: {{__kmpc|__tgt}}
#ifdef CK12


// CK12-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0

// CK12: [[SIZE09:@.+]] = private {{.*}}constant [1 x i64] [i64 {{8|4}}]
// Map types: OMP_MAP_FROM | OMP_MAP_TARGET_PARAM | OMP_MAP_IMPLICIT = 546
// CK12: [[MTYPE09:@.+]] = private {{.*}}constant [1 x i64] [i64 546]

// CK12-LABEL: explicit_maps_single{{.*}}(
void explicit_maps_single (){
  int *pa;

  // CK12-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE09]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE09]]{{.+}}, i8** null)
  // CK12-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK12-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK12-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK12-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK12-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i32***
  // CK12-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32***
  // CK12-DAG: store i32** [[VAR0:%.+]], i32*** [[CBP0]]
  // CK12-DAG: store i32** [[VAR0]], i32*** [[CP0]]

  // CK12: call void [[CALL09:@.+]](i32** {{[^,]+}})
  #pragma omp target defaultmap(from: pointer)
  {
    pa[50]++;
  }
}

  // CK12: define {{.+}}[[CALL09]]
#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK13 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK13
// RUN: %clang_cc1 -DCK13 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK13
// RUN: %clang_cc1 -DCK13 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK13
// RUN: %clang_cc1 -DCK13 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK13

// RUN: %clang_cc1 -DCK13 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY8 %s
// RUN: %clang_cc1 -DCK13 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY8 %s
// RUN: %clang_cc1 -DCK13 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY8 %s
// RUN: %clang_cc1 -DCK13 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY8 %s
// SIMD-ONLY8-NOT: {{__kmpc|__tgt}}
#ifdef CK13


// CK13-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0

// CK13: [[SIZE09:@.+]] = private {{.*}}constant [1 x i64] [i64 {{8|4}}]
// Map types: OMP_MAP_TO | OMP_MAP_FROM | OMP_MAP_TARGET_PARAM | OMP_MAP_IMPLICIT = 547
// CK13: [[MTYPE09:@.+]] = private {{.*}}constant [1 x i64] [i64 547]

// CK13-LABEL: explicit_maps_single{{.*}}(
void explicit_maps_single (){
  int *pa;

  // CK13-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE09]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE09]]{{.+}}, i8** null)
  // CK13-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK13-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK13-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK13-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK13-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i32***
  // CK13-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32***
  // CK13-DAG: store i32** [[VAR0:%.+]], i32*** [[CBP0]]
  // CK13-DAG: store i32** [[VAR0]], i32*** [[CP0]]

  // CK13: call void [[CALL09:@.+]](i32** {{[^,]+}})
  #pragma omp target defaultmap(tofrom: pointer)
  {
    pa[50]++;
  }
}

  // CK13: define {{.+}}[[CALL09]]
#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK14 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK14
// RUN: %clang_cc1 -DCK14 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK14
// RUN: %clang_cc1 -DCK14 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK14
// RUN: %clang_cc1 -DCK14 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK14

// RUN: %clang_cc1 -DCK14 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY8 %s
// RUN: %clang_cc1 -DCK14 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY8 %s
// RUN: %clang_cc1 -DCK14 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY8 %s
// RUN: %clang_cc1 -DCK14 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY8 %s
// SIMD-ONLY8-NOT: {{__kmpc|__tgt}}
#ifdef CK14


// CK14-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0

// CK14: [[SIZE09:@.+]] = private {{.*}}constant [1 x i64] zeroinitializer
// Map types: OMP_MAP_TARGET_PARAM | OMP_MAP_IMPLICIT = 544
// CK14: [[MTYPE09:@.+]] = private {{.*}}constant [1 x i64] [i64 544]

// CK14-LABEL: explicit_maps_single{{.*}}(
void explicit_maps_single (){
  int *pa;

  // CK14-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE09]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE09]]{{.+}}, i8** null)
  // CK14-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK14-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK14-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK14-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK14-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i32**
  // CK14-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK14-DAG: store i32* [[VAR0:%.+]], i32** [[CBP0]]
  // CK14-DAG: store i32* [[VAR0]], i32** [[CP0]]

  // CK14: call void [[CALL09:@.+]](i32* {{[^,]+}})
  #pragma omp target defaultmap(firstprivate: pointer)
  {
    pa[50]++;
  }
}

  // CK14: define {{.+}}[[CALL09]]
#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK15 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK15
// RUN: %clang_cc1 -DCK15 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK15
// RUN: %clang_cc1 -DCK15 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK15
// RUN: %clang_cc1 -DCK15 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK15

// RUN: %clang_cc1 -DCK15 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY12 %s
// RUN: %clang_cc1 -DCK15 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY12 %s
// RUN: %clang_cc1 -DCK15 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY12 %s
// RUN: %clang_cc1 -DCK15 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY12 %s
// SIMD-ONLY12-NOT: {{__kmpc|__tgt}}
#ifdef CK15

// CK15-LABEL: @.__omp_offloading_{{.*}}implicit_maps_variable_length_array{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0

// We don't have a constant map size for VLAs.
// Map types:
//  - OMP_MAP_LITERAL + OMP_MAP_TARGET_PARAM + OMP_MAP_IMPLICIT = 800 (vla size)
//  - OMP_MAP_LITERAL + OMP_MAP_TARGET_PARAM + OMP_MAP_IMPLICIT = 800 (vla size)
//  - OMP_MAP_TARGET_PARAM | OMP_MAP_IMPLICIT = 544
// CK15-DAG: [[TYPES:@.+]] = {{.+}}constant [3 x i64] [i64 800, i64 800, i64 544]

// CK15-LABEL: implicit_maps_variable_length_array{{.*}}(
void implicit_maps_variable_length_array (int a){
  double vla[2][a];

  // CK15-DAG: call i32 @__tgt_target_mapper(i64 {{.+}}, i8* {{.+}}, i32 3, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], i64* [[SGEP:%[^,]+]], {{.+}}[[TYPES]]{{.+}}, i8** null)
  // CK15-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK15-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK15-DAG: [[SGEP]] = getelementptr inbounds {{.+}}[[SS:%[^,]+]], i32 0, i32 0

  // CK15-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK15-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK15-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[SS]], i32 0, i32 0
  // CK15-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i[[sz:64|32]]*
  // CK15-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i[[sz]]*
  // CK15-DAG: store i[[sz]] 2, i[[sz]]* [[CBP0]]
  // CK15-DAG: store i[[sz]] 2, i[[sz]]* [[CP0]]
  // CK15-DAG: store i64 {{8|4}}, i64* [[S0]],

  // CK15-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 1
  // CK15-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 1
  // CK15-DAG: [[S1:%.+]] = getelementptr inbounds {{.+}}[[SS]], i32 0, i32 1
  // CK15-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to i[[sz]]*
  // CK15-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i[[sz]]*
  // CK15-DAG: store i[[sz]] [[VAL:%.+]], i[[sz]]* [[CBP1]]
  // CK15-DAG: store i[[sz]] [[VAL]], i[[sz]]* [[CP1]]
  // CK15-DAG: store i64 {{8|4}}, i64* [[S1]],

  // CK15-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 2
  // CK15-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 2
  // CK15-DAG: [[S2:%.+]] = getelementptr inbounds {{.+}}[[SS]], i32 0, i32 2
  // CK15-DAG: [[CBP2:%.+]] = bitcast i8** [[BP2]] to double**
  // CK15-DAG: [[CP2:%.+]] = bitcast i8** [[P2]] to double**
  // CK15-DAG: store double* [[DECL:%.+]], double** [[CBP2]]
  // CK15-DAG: store double* [[DECL]], double** [[CP2]]
  // CK15-DAG: store i64 [[VALS2:%.+]], i64* [[S2]],
  // CK15-DAG: [[VALS2]] = {{mul nuw i64 %.+, 8|sext i32 %.+ to i64}}

  // CK15: call void [[KERNEL:@.+]](i[[sz]] {{.+}}, i[[sz]] {{.+}}, double* [[DECL]])
  #pragma omp target defaultmap(alloc: aggregate)
  {
    vla[1][3] += 1.0;
  }
}

// CK15: define internal void [[KERNEL]](i[[sz]] [[VLA0:%.+]], i[[sz]] [[VLA1:%.+]], double* {{.*}}[[ARG:%.+]])
// CK15: [[ADDR0:%.+]] = alloca i[[sz]],
// CK15: [[ADDR1:%.+]] = alloca i[[sz]],
// CK15: [[ADDR2:%.+]] = alloca double*,
// CK15: store i[[sz]] [[VLA0]], i[[sz]]* [[ADDR0]],
// CK15: store i[[sz]] [[VLA1]], i[[sz]]* [[ADDR1]],
// CK15: store double* [[ARG]], double** [[ADDR2]],
// CK15: {{.+}} = load i[[sz]],  i[[sz]]* [[ADDR0]],
// CK15: {{.+}} = load i[[sz]],  i[[sz]]* [[ADDR1]],
// CK15: [[REF:%.+]] = load double*, double** [[ADDR2]],
// CK15: {{.+}} = getelementptr inbounds double, double* [[REF]], i[[sz]] %{{.+}}
#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK16 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK16
// RUN: %clang_cc1 -DCK16 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK16
// RUN: %clang_cc1 -DCK16 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK16
// RUN: %clang_cc1 -DCK16 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK16

// RUN: %clang_cc1 -DCK16 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY16 %s
// RUN: %clang_cc1 -DCK16 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY16 %s
// RUN: %clang_cc1 -DCK16 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY16 %s
// RUN: %clang_cc1 -DCK16 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY16 %s
// SIMD-ONLY16-NOT: {{__kmpc|__tgt}}
#ifdef CK16

// CK16-DAG: [[ST:%.+]] = type { i32, double }
// CK16-LABEL: @.__omp_offloading_{{.*}}implicit_maps_struct{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK16-DAG: [[SIZES:@.+]] = {{.+}}constant [1 x i64] [i64 {{16|12}}]
// Map types: OMP_MAP_TARGET_PARAM | OMP_MAP_IMPLICIT = 544
// CK16-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i64] [i64 544]

class SSS {
public:
  int a;
  double b;
};

// CK16-LABEL: implicit_maps_struct{{.*}}(
void implicit_maps_struct (int a){
  SSS s = {a, (double)a};

  // CK16-DAG: call i32 @__tgt_target_mapper(i64 {{.+}}, i8* {{.+}}, i32 1, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZES]]{{.+}}, {{.+}}[[TYPES]]{{.+}}, i8** null)
  // CK16-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK16-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK16-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK16-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK16-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to [[ST]]**
  // CK16-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to [[ST]]**
  // CK16-DAG: store [[ST]]* [[DECL:%.+]], [[ST]]** [[CBP1]]
  // CK16-DAG: store [[ST]]* [[DECL]], [[ST]]** [[CP1]]

  // CK16: call void [[KERNEL:@.+]]([[ST]]* [[DECL]])
  #pragma omp target defaultmap(alloc: aggregate)
  {
    s.a += 1;
    s.b += 1.0;
  }
}

// CK16: define internal void [[KERNEL]]([[ST]]* {{.+}}[[ARG:%.+]])
// CK16: [[ADDR:%.+]] = alloca [[ST]]*,
// CK16: store [[ST]]* [[ARG]], [[ST]]** [[ADDR]],
// CK16: [[REF:%.+]] = load [[ST]]*, [[ST]]** [[ADDR]],
// CK16: {{.+}} = getelementptr inbounds [[ST]], [[ST]]* [[REF]], i32 0, i32 0
#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK17 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK17
// RUN: %clang_cc1 -DCK17 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK17
// RUN: %clang_cc1 -DCK17 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK17
// RUN: %clang_cc1 -DCK17 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK17

// RUN: %clang_cc1 -DCK17 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY16 %s
// RUN: %clang_cc1 -DCK17 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY16 %s
// RUN: %clang_cc1 -DCK17 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY16 %s
// RUN: %clang_cc1 -DCK17 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY16 %s
// SIMD-ONLY16-NOT: {{__kmpc|__tgt}}
#ifdef CK17

// CK17-DAG: [[ST:%.+]] = type { i32, double }
// CK17-LABEL: @.__omp_offloading_{{.*}}implicit_maps_struct{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK17-DAG: [[SIZES:@.+]] = {{.+}}constant [1 x i64] [i64 {{16|12}}]
// Map types: OMP_MAP_TO | OMP_MAP_TARGET_PARAM | OMP_MAP_IMPLICIT = 545
// CK17-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i64] [i64 545]

class SSS {
public:
  int a;
  double b;
};

// CK17-LABEL: implicit_maps_struct{{.*}}(
void implicit_maps_struct (int a){
  SSS s = {a, (double)a};

  // CK17-DAG: call i32 @__tgt_target_mapper(i64 {{.+}}, i8* {{.+}}, i32 1, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZES]]{{.+}}, {{.+}}[[TYPES]]{{.+}}, i8** null)
  // CK17-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK17-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK17-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK17-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK17-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to [[ST]]**
  // CK17-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to [[ST]]**
  // CK17-DAG: store [[ST]]* [[DECL:%.+]], [[ST]]** [[CBP1]]
  // CK17-DAG: store [[ST]]* [[DECL]], [[ST]]** [[CP1]]

  // CK17: call void [[KERNEL:@.+]]([[ST]]* [[DECL]])
  #pragma omp target defaultmap(to: aggregate)
  {
    s.a += 1;
    s.b += 1.0;
  }
}

// CK17: define internal void [[KERNEL]]([[ST]]* {{.+}}[[ARG:%.+]])
// CK17: [[ADDR:%.+]] = alloca [[ST]]*,
// CK17: store [[ST]]* [[ARG]], [[ST]]** [[ADDR]],
// CK17: [[REF:%.+]] = load [[ST]]*, [[ST]]** [[ADDR]],
// CK17: {{.+}} = getelementptr inbounds [[ST]], [[ST]]* [[REF]], i32 0, i32 0
#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK18 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK18
// RUN: %clang_cc1 -DCK18 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK18
// RUN: %clang_cc1 -DCK18 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK18
// RUN: %clang_cc1 -DCK18 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK18

// RUN: %clang_cc1 -DCK18 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY16 %s
// RUN: %clang_cc1 -DCK18 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY16 %s
// RUN: %clang_cc1 -DCK18 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY16 %s
// RUN: %clang_cc1 -DCK18 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY16 %s
// SIMD-ONLY16-NOT: {{__kmpc|__tgt}}
#ifdef CK18

// CK18-DAG: [[ST:%.+]] = type { i32, double }
// CK18-LABEL: @.__omp_offloading_{{.*}}implicit_maps_struct{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK18-DAG: [[SIZES:@.+]] = {{.+}}constant [1 x i64] [i64 {{16|12}}]
// Map types: OMP_MAP_FROM | OMP_MAP_TARGET_PARAM | OMP_MAP_IMPLICIT = 546
// CK18-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i64] [i64 546]

class SSS {
public:
  int a;
  double b;
};

// CK18-LABEL: implicit_maps_struct{{.*}}(
void implicit_maps_struct (int a){
  SSS s = {a, (double)a};

  // CK18-DAG: call i32 @__tgt_target_mapper(i64 {{.+}}, i8* {{.+}}, i32 1, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZES]]{{.+}}, {{.+}}[[TYPES]]{{.+}}, i8** null)
  // CK18-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK18-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK18-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK18-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK18-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to [[ST]]**
  // CK18-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to [[ST]]**
  // CK18-DAG: store [[ST]]* [[DECL:%.+]], [[ST]]** [[CBP1]]
  // CK18-DAG: store [[ST]]* [[DECL]], [[ST]]** [[CP1]]

  // CK18: call void [[KERNEL:@.+]]([[ST]]* [[DECL]])
  #pragma omp target defaultmap(from: aggregate)
  {
    s.a += 1;
    s.b += 1.0;
  }
}

// CK18: define internal void [[KERNEL]]([[ST]]* {{.+}}[[ARG:%.+]])
// CK18: [[ADDR:%.+]] = alloca [[ST]]*,
// CK18: store [[ST]]* [[ARG]], [[ST]]** [[ADDR]],
// CK18: [[REF:%.+]] = load [[ST]]*, [[ST]]** [[ADDR]],
// CK18: {{.+}} = getelementptr inbounds [[ST]], [[ST]]* [[REF]], i32 0, i32 0
#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK19 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK19
// RUN: %clang_cc1 -DCK19 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK19
// RUN: %clang_cc1 -DCK19 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK19
// RUN: %clang_cc1 -DCK19 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK19

// RUN: %clang_cc1 -DCK19 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY16 %s
// RUN: %clang_cc1 -DCK19 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY16 %s
// RUN: %clang_cc1 -DCK19 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY16 %s
// RUN: %clang_cc1 -DCK19 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY16 %s
// SIMD-ONLY16-NOT: {{__kmpc|__tgt}}
#ifdef CK19

// CK19-DAG: [[ST:%.+]] = type { i32, double }
// CK19-LABEL: @.__omp_offloading_{{.*}}implicit_maps_struct{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19-DAG: [[SIZES:@.+]] = {{.+}}constant [1 x i64] [i64 {{16|12}}]
// Map types: OMP_MAP_TO | OMP_MAP_FROM | OMP_MAP_TARGET_PARAM | OMP_MAP_IMPLICIT = 547
// CK19-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i64] [i64 547]

class SSS {
public:
  int a;
  double b;
};

// CK19-LABEL: implicit_maps_struct{{.*}}(
void implicit_maps_struct (int a){
  SSS s = {a, (double)a};

  // CK19-DAG: call i32 @__tgt_target_mapper(i64 {{.+}}, i8* {{.+}}, i32 1, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZES]]{{.+}}, {{.+}}[[TYPES]]{{.+}}, i8** null)
  // CK19-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK19-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK19-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK19-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK19-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to [[ST]]**
  // CK19-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to [[ST]]**
  // CK19-DAG: store [[ST]]* [[DECL:%.+]], [[ST]]** [[CBP1]]
  // CK19-DAG: store [[ST]]* [[DECL]], [[ST]]** [[CP1]]

  // CK19: call void [[KERNEL:@.+]]([[ST]]* [[DECL]])
  #pragma omp target defaultmap(tofrom: aggregate)
  {
    s.a += 1;
    s.b += 1.0;
  }
}

// CK19: define internal void [[KERNEL]]([[ST]]* {{.+}}[[ARG:%.+]])
// CK19: [[ADDR:%.+]] = alloca [[ST]]*,
// CK19: store [[ST]]* [[ARG]], [[ST]]** [[ADDR]],
// CK19: [[REF:%.+]] = load [[ST]]*, [[ST]]** [[ADDR]],
// CK19: {{.+}} = getelementptr inbounds [[ST]], [[ST]]* [[REF]], i32 0, i32 0
#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK20 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK20 --check-prefix CK20-64
// RUN: %clang_cc1 -DCK20 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK20  --check-prefix CK20-64
// RUN: %clang_cc1 -DCK20 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK20  --check-prefix CK20-32
// RUN: %clang_cc1 -DCK20 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK20  --check-prefix CK20-32

// RUN: %clang_cc1 -DCK20 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY6 %s
// RUN: %clang_cc1 -DCK20 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY6 %s
// RUN: %clang_cc1 -DCK20 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY6 %s
// RUN: %clang_cc1 -DCK20 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY6 %s
// SIMD-ONLY6-NOT: {{__kmpc|__tgt}}
#ifdef CK20

// For a 32-bit targets, the value doesn't fit the size of the pointer,
// therefore it is passed by reference with a map 'to' specification.

// CK20-LABEL: @.__omp_offloading_{{.*}}implicit_maps_double{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0

// CK20-DAG: [[SIZES:@.+]] = {{.+}}constant [1 x i64] [i64 8]
// Map types: OMP_MAP_LITERAL | OMP_MAP_TARGET_PARAM | OMP_MAP_IMPLICIT = 800
// CK20-64-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i64] [i64 800]
// Map types: OMP_MAP_TO | OMP_MAP_PRIVATE | OMP_MAP_TARGET_PARAM | OMP_MAP_IMPLICIT = 673
// CK20-32-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i64] [i64 673]

// CK20-LABEL: implicit_maps_double{{.*}}(
void implicit_maps_double (int a){
  double d = (double)a;

  // CK20-DAG: call i32 @__tgt_target_mapper(i64 {{.+}}, i8* {{.+}}, i32 1, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZES]]{{.+}}, {{.+}}[[TYPES]]{{.+}}, i8** null)
  // CK20-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK20-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK20-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK20-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0

  // CK20-64-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to i[[sz:64|32]]*
  // CK20-64-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i[[sz]]*
  // CK20-64-DAG: store i[[sz]] [[VAL:%[^,]+]], i[[sz]]* [[CBP1]]
  // CK20-64-DAG: store i[[sz]] [[VAL]], i[[sz]]* [[CP1]]
  // CK20-64-DAG: [[VAL]] = load i[[sz]], i[[sz]]* [[ADDR:%.+]],
  // CK20-64-64-DAG: [[CADDR:%.+]] = bitcast i[[sz]]* [[ADDR]] to double*
  // CK20-64-64-DAG: store double {{.+}}, double* [[CADDR]],

  // CK20-32-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to double**
  // CK20-32-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to double**
  // CK20-32-DAG: store double* [[DECL:%[^,]+]], double** [[CBP1]]
  // CK20-32-DAG: store double* [[DECL]], double** [[CP1]]

  // CK20-64: call void [[KERNEL:@.+]](i[[sz]] [[VAL]])
  // CK20-32: call void [[KERNEL:@.+]](double* [[DECL]])
  #pragma omp target defaultmap(default: scalar)
  {
    d += 1.0;
  }
}

// CK20-64: define internal void [[KERNEL]](i[[sz]] [[ARG:%.+]])
// CK20-64: [[ADDR:%.+]] = alloca i[[sz]],
// CK20-64: store i[[sz]] [[ARG]], i[[sz]]* [[ADDR]],
// CK20-64: [[CADDR:%.+]] = bitcast i64* [[ADDR]] to double*
// CK20-64: {{.+}} = load double, double* [[CADDR]],

// CK20-32: define internal void [[KERNEL]](double* {{.+}}[[ARG:%.+]])
// CK20-32: [[ADDR:%.+]] = alloca double*,
// CK20-32: store double* [[ARG]], double** [[ADDR]],
// CK20-32: [[REF:%.+]] = load double*, double** [[ADDR]],
// CK20-32: {{.+}} = load double, double* [[REF]],

#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK21 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK21
// RUN: %clang_cc1 -DCK21 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK21
// RUN: %clang_cc1 -DCK21 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK21
// RUN: %clang_cc1 -DCK21 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK21

// RUN: %clang_cc1 -DCK21 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY16 %s
// RUN: %clang_cc1 -DCK21 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY16 %s
// RUN: %clang_cc1 -DCK21 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY16 %s
// RUN: %clang_cc1 -DCK21 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY16 %s
// SIMD-ONLY16-NOT: {{__kmpc|__tgt}}
#ifdef CK21

// CK21-DAG: [[ST:%.+]] = type { i32, double }
// CK21-LABEL: @.__omp_offloading_{{.*}}implicit_maps_struct{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK21-DAG: [[SIZES:@.+]] = {{.+}}constant [1 x i64] [i64 {{16|12}}]
// Map types: OMP_MAP_TO + OMP_MAP_FROM + OMP_MAP_TARGET_PARAM | OMP_MAP_IMPLICIT = 547
// CK21-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i64] [i64 547]

class SSS {
public:
  int a;
  double b;
};

// CK21-LABEL: implicit_maps_struct{{.*}}(
void implicit_maps_struct (int a){
  SSS s = {a, (double)a};

  // CK21-DAG: call i32 @__tgt_target_mapper(i64 {{.+}}, i8* {{.+}}, i32 1, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZES]]{{.+}}, {{.+}}[[TYPES]]{{.+}}, i8** null)
  // CK21-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK21-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK21-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK21-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK21-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to [[ST]]**
  // CK21-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to [[ST]]**
  // CK21-DAG: store [[ST]]* [[DECL:%.+]], [[ST]]** [[CBP1]]
  // CK21-DAG: store [[ST]]* [[DECL]], [[ST]]** [[CP1]]

  // CK21: call void [[KERNEL:@.+]]([[ST]]* [[DECL]])
  #pragma omp target defaultmap(default: aggregate)
  {
    s.a += 1;
    s.b += 1.0;
  }
}

// CK21: define internal void [[KERNEL]]([[ST]]* {{.+}}[[ARG:%.+]])
// CK21: [[ADDR:%.+]] = alloca [[ST]]*,
// CK21: store [[ST]]* [[ARG]], [[ST]]** [[ADDR]],
// CK21: [[REF:%.+]] = load [[ST]]*, [[ST]]** [[ADDR]],
// CK21: {{.+}} = getelementptr inbounds [[ST]], [[ST]]* [[REF]], i32 0, i32 0
#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK22 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK22
// RUN: %clang_cc1 -DCK22 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK22
// RUN: %clang_cc1 -DCK22 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK22
// RUN: %clang_cc1 -DCK22 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK22

// RUN: %clang_cc1 -DCK22 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY9 %s
// RUN: %clang_cc1 -DCK22 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY9 %s
// RUN: %clang_cc1 -DCK22 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY9 %s
// RUN: %clang_cc1 -DCK22 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY9 %s
// SIMD-ONLY9-NOT: {{__kmpc|__tgt}}
#ifdef CK22

// CK22-LABEL: @.__omp_offloading_{{.*}}implicit_maps_pointer{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0

// CK22-DAG: [[SIZES:@.+]] = {{.+}}constant [1 x i64] zeroinitializer
// Map types: OMP_MAP_TARGET_PARAM | OMP_MAP_IMPLICIT = 544
// CK22-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i64] [i64 544]

// CK22-LABEL: implicit_maps_pointer{{.*}}(
void implicit_maps_pointer (){
  double *ddyn;

  // CK22-DAG: call i32 @__tgt_target_mapper(i64 {{.+}}, i8* {{.+}}, i32 1, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZES]]{{.+}}, {{.+}}[[TYPES]]{{.+}}, i8** null)
  // CK22-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK22-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK22-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK22-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK22-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to double**
  // CK22-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to double**
  // CK22-DAG: store double* [[PTR:%[^,]+]], double** [[CBP1]]
  // CK22-DAG: store double* [[PTR]], double** [[CP1]]

  // CK22: call void [[KERNEL:@.+]](double* [[PTR]])
  #pragma omp target defaultmap(default: pointer)
  {
    ddyn[0] += 1.0;
    ddyn[1] += 1.0;
  }
}

// CK22: define internal void [[KERNEL]](double* {{.*}}[[ARG:%.+]])
// CK22: [[ADDR:%.+]] = alloca double*,
// CK22: store double* [[ARG]], double** [[ADDR]],
// CK22: [[REF:%.+]] = load double*, double** [[ADDR]],
// CK22: {{.+}} = getelementptr inbounds double, double* [[REF]], i{{64|32}} 0

#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK23 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK23 --check-prefix CK23-64
// RUN: %clang_cc1 -DCK23 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK23 --check-prefix CK23-64
// RUN: %clang_cc1 -DCK23 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK23 --check-prefix CK23-32
// RUN: %clang_cc1 -DCK23 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK23 --check-prefix CK23-32

// RUN: %clang_cc1 -DCK23 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK23 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK23 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK23 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
#ifdef CK23

double *g;

// CK23: @g = global double*
// CK23: [[SIZES00:@.+]] = {{.+}}constant [1 x i[[sz:64|32]]] [i{{64|32}} {{8|4}}]
// CK23: [[TYPES00:@.+]] = {{.+}}constant [1 x i64] [i64 288]

// CK23: [[SIZES01:@.+]] = {{.+}}constant [1 x i[[sz]]] [i[[sz]] {{8|4}}]
// CK23: [[TYPES01:@.+]] = {{.+}}constant [1 x i64] [i64 288]

// CK23: [[SIZES02:@.+]] = {{.+}}constant [1 x i[[sz]]] [i[[sz]] {{8|4}}]
// CK23: [[TYPES02:@.+]] = {{.+}}constant [1 x i64] [i64 288]

// CK23: [[SIZES03:@.+]] = {{.+}}constant [1 x i[[sz]]] [i[[sz]] {{8|4}}]
// CK23: [[TYPES03:@.+]] = {{.+}}constant [1 x i64] [i64 288]

// CK23: [[SIZES04:@.+]] = {{.+}}constant [1 x i[[sz]]] [i[[sz]] {{8|4}}]
// CK23: [[TYPES04:@.+]] = {{.+}}constant [1 x i64] [i64 288]

// CK23: [[SIZES05:@.+]] = {{.+}}constant [1 x i[[sz]]] [i[[sz]] {{8|4}}]
// CK23: [[TYPES05:@.+]] = {{.+}}constant [1 x i64] [i64 288]

// CK23: [[SIZES06:@.+]] = {{.+}}constant [2 x i[[sz]]] [i[[sz]] {{8|4}}, i[[sz]] {{8|4}}]
// CK23: [[TYPES06:@.+]] = {{.+}}constant [2 x i64] [i64 288, i64 288]

// CK23-LABEL: @_Z3foo{{.*}}(
template<typename T>
void foo(float *&lr, T *&tr) {
  float *l;
  T *t;

  // CK23-DAG: call i32 @__tgt_target_mapper(i64 {{.+}}, i8* {{.+}}, i32 1, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZES00]]{{.+}}, {{.+}}[[TYPES00]]{{.+}}, i8** null)
  // CK23-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK23-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK23-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK23-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK23-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to double**
  // CK23-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to double**
  // CK23-DAG: store double* [[VAL:%.+]], double** [[CBP1]]
  // CK23-DAG: store double* [[VAL]], double** [[CP1]]
  // CK23-DAG: [[VAL]] = load double*, double** [[ADDR:@g]],

  // CK23: call void [[KERNEL:@.+]](double* [[VAL]])
  #pragma omp target is_device_ptr(g) defaultmap(none:pointer)
  {
    ++g;
  }

  // CK23-DAG: call i32 @__tgt_target_mapper(i64 {{.+}}, i8* {{.+}}, i32 1, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZES01]]{{.+}}, {{.+}}[[TYPES01]]{{.+}}, i8** null)
  // CK23-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK23-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK23-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK23-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK23-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to float**
  // CK23-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to float**
  // CK23-DAG: store float* [[VAL:%.+]], float** [[CBP1]]
  // CK23-DAG: store float* [[VAL]], float** [[CP1]]
  // CK23-DAG: [[VAL]] = load float*, float** [[ADDR:%.+]],

  // CK23: call void [[KERNEL:@.+]](float* [[VAL]])
  #pragma omp target is_device_ptr(l) defaultmap(none:pointer)
  {
    ++l;
  }

  // CK23-DAG: call i32 @__tgt_target_mapper(i64 {{.+}}, i8* {{.+}}, i32 1, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZES02]]{{.+}}, {{.+}}[[TYPES02]]{{.+}}, i8** null)
  // CK23-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK23-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK23-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK23-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK23-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to i32**
  // CK23-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i32**
  // CK23-DAG: store i32* [[VAL:%.+]], i32** [[CBP1]]
  // CK23-DAG: store i32* [[VAL]], i32** [[CP1]]
  // CK23-DAG: [[VAL]] = load i32*, i32** [[ADDR:%.+]],

  // CK23: call void [[KERNEL:@.+]](i32* [[VAL]])
  #pragma omp target is_device_ptr(t) defaultmap(none:pointer)
  {
    ++t;
  }

  // CK23-DAG: call i32 @__tgt_target_mapper(i64 {{.+}}, i8* {{.+}}, i32 1, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZES03]]{{.+}}, {{.+}}[[TYPES03]]{{.+}}, i8** null)
  // CK23-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK23-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK23-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK23-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK23-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to float**
  // CK23-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to float**
  // CK23-DAG: store float* [[VAL:%.+]], float** [[CBP1]]
  // CK23-DAG: store float* [[VAL]], float** [[CP1]]
  // CK23-DAG: [[VAL]] = load float*, float** [[ADDR:%.+]],
  // CK23-DAG: [[ADDR]] = load float**, float*** [[ADDR2:%.+]],

  // CK23: call void [[KERNEL:@.+]](float* [[VAL]])
  #pragma omp target is_device_ptr(lr) defaultmap(none:pointer)
  {
    ++lr;
  }

  // CK23-DAG: call i32 @__tgt_target_mapper(i64 {{.+}}, i8* {{.+}}, i32 1, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZES04]]{{.+}}, {{.+}}[[TYPES04]]{{.+}}, i8** null)
  // CK23-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK23-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK23-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK23-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK23-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to i32**
  // CK23-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i32**
  // CK23-DAG: store i32* [[VAL:%.+]], i32** [[CBP1]]
  // CK23-DAG: store i32* [[VAL]], i32** [[CP1]]
  // CK23-DAG: [[VAL]] = load i32*, i32** [[ADDR:%.+]],
  // CK23-DAG: [[ADDR]] = load i32**, i32*** [[ADDR2:%.+]],

  // CK23: call void [[KERNEL:@.+]](i32* [[VAL]])
  #pragma omp target is_device_ptr(tr) defaultmap(none:pointer)
  {
    ++tr;
  }

  // CK23-DAG: call i32 @__tgt_target_mapper(i64 {{.+}}, i8* {{.+}}, i32 1, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZES05]]{{.+}}, {{.+}}[[TYPES05]]{{.+}}, i8** null)
  // CK23-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK23-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK23-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK23-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK23-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to i32**
  // CK23-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i32**
  // CK23-DAG: store i32* [[VAL:%.+]], i32** [[CBP1]]
  // CK23-DAG: store i32* [[VAL]], i32** [[CP1]]
  // CK23-DAG: [[VAL]] = load i32*, i32** [[ADDR:%.+]],
  // CK23-DAG: [[ADDR]] = load i32**, i32*** [[ADDR2:%.+]],

  // CK23: call void [[KERNEL:@.+]](i32* [[VAL]])
  #pragma omp target is_device_ptr(tr,lr) defaultmap(none:pointer)
  {
    ++tr;
  }

  // CK23-DAG: call i32 @__tgt_target_mapper(i64 {{.+}}, i8* {{.+}}, i32 2, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZES06]]{{.+}}, {{.+}}[[TYPES06]]{{.+}}, i8** null)
  // CK23-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK23-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK23-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK23-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK23-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to i32**
  // CK23-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i32**
  // CK23-DAG: store i32* [[VAL:%.+]], i32** [[CBP1]]
  // CK23-DAG: store i32* [[VAL]], i32** [[CP1]]
  // CK23-DAG: [[VAL]] = load i32*, i32** [[ADDR:%.+]],
  // CK23-DAG: [[ADDR]] = load i32**, i32*** [[ADDR2:%.+]],

  // CK23-DAG: [[_BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 1
  // CK23-DAG: [[_P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 1
  // CK23-DAG: [[_CBP1:%.+]] = bitcast i8** [[_BP1]] to float**
  // CK23-DAG: [[_CP1:%.+]] = bitcast i8** [[_P1]] to float**
  // CK23-DAG: store float* [[_VAL:%.+]], float** [[_CBP1]]
  // CK23-DAG: store float* [[_VAL]], float** [[_CP1]]
  // CK23-DAG: [[_VAL]] = load float*, float** [[_ADDR:%.+]],
  // CK23-DAG: [[_ADDR]] = load float**, float*** [[_ADDR2:%.+]],

  // CK23: call void [[KERNEL:@.+]](i32* [[VAL]], float* [[_VAL]])
  #pragma omp target is_device_ptr(tr,lr) defaultmap(none:pointer)
  {
    ++tr,++lr;
  }
}

void bar(float *&a, int *&b) {
  foo<int>(a,b);
}

#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK24 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK24 --check-prefix CK24-64
// RUN: %clang_cc1 -DCK24 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK24 --check-prefix CK24-64
// RUN: %clang_cc1 -DCK24 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK24 --check-prefix CK24-32
// RUN: %clang_cc1 -DCK24 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK24 --check-prefix CK24-32

// RUN: %clang_cc1 -DCK24 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY18 %s
// RUN: %clang_cc1 -DCK24 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY18 %s
// RUN: %clang_cc1 -DCK24 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY18 %s
// RUN: %clang_cc1 -DCK24 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY18 %s
// SIMD-ONLY18-NOT: {{__kmpc|__tgt}}
#ifdef CK24

// CK24-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK24: [[SIZE00:@.+]] = private {{.*}}constant [1 x i[[Z:64|32]]] [i[[Z:64|32]] 4]
// CK24: [[MTYPE00:@.+]] = private {{.*}}constant [1 x i64] [i64 1059]

// CK24-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK24: [[SIZE01:@.+]] = private {{.*}}constant [1 x i[[Z:64|32]]] [i[[Z:64|32]] 4]
// CK24: [[MTYPE01:@.+]] = private {{.*}}constant [1 x i64] [i64 1063]

// CK24-LABEL: explicit_maps_single{{.*}}(
void explicit_maps_single (int ii){
  // Map of a scalar.
  int a = ii;

  // Close.
  // Region 00
  // CK24-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE00]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE00]]{{.+}}, i8** null)
  // CK24-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK24-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK24-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK24-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK24-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i32**
  // CK24-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK24-DAG: store i32* [[VAR0:%.+]], i32** [[CBP0]]
  // CK24-DAG: store i32* [[VAR0]], i32** [[CP0]]

  // CK24: call void [[CALL00:@.+]](i32* {{[^,]+}})
  #pragma omp target map(close, tofrom: a) defaultmap(none:scalar)
  {
   a++;
  }

  // Always Close.
  // Region 01
  // CK24-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE01]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE01]]{{.+}}, i8** null)
  // CK24-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK24-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK24-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK24-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK24-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i32**
  // CK24-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK24-DAG: store i32* [[VAR0:%.+]], i32** [[CBP0]]
  // CK24-DAG: store i32* [[VAR0]], i32** [[CP0]]

  // CK24: call void [[CALL01:@.+]](i32* {{[^,]+}})
  #pragma omp target map(always close tofrom: a) defaultmap(none:scalar)
  {
   a++;
  }
}
// CK24: define {{.+}}[[CALL00]]
// CK24: define {{.+}}[[CALL01]]

#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK25 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK25 --check-prefix CK25-64
// RUN: %clang_cc1 -DCK25 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK25 --check-prefix CK25-64
// RUN: %clang_cc1 -DCK25 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK25 --check-prefix CK25-32
// RUN: %clang_cc1 -DCK25 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK25 --check-prefix CK25-32

// RUN: %clang_cc1 -DCK25 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY18 %s
// RUN: %clang_cc1 -DCK25 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY18 %s
// RUN: %clang_cc1 -DCK25 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY18 %s
// RUN: %clang_cc1 -DCK25 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY18 %s
// SIMD-ONLY18-NOT: {{__kmpc|__tgt}}
#ifdef CK25

extern int x;
#pragma omp declare target to(x)

// CK25-LABEL: @.__omp_offloading_{{.*}}declare_target_to{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0

void declare_target_to()
{
  // CK25: [[C:%.+]] = load i32, i32* @x
  // CK25: [[INC:%.+]] = add nsw i32 [[C]], 1
  // CK25: store i32 [[INC]], i32* @x
  // CK25: ret void
  #pragma omp target defaultmap(none : scalar)
  {
    x++;
  }
}

#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK26 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK26 --check-prefix CK26-64
// RUN: %clang_cc1 -DCK26 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK26 --check-prefix CK26-64
// RUN: %clang_cc1 -DCK26 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK26 --check-prefix CK26-32
// RUN: %clang_cc1 -DCK26 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK26 --check-prefix CK26-32

// RUN: %clang_cc1 -DCK26 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY18 %s
// RUN: %clang_cc1 -DCK26 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY18 %s
// RUN: %clang_cc1 -DCK26 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY18 %s
// RUN: %clang_cc1 -DCK26 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY18 %s
// SIMD-ONLY18-NOT: {{__kmpc|__tgt}}
#ifdef CK26

// CK26-DAG: [[SIZES:@.+]] = {{.+}}constant [3 x i64] [i64 4096, i64 4, i64 {{.+}}]
// Map types: OMP_MAP_TO | MAP_ALWAYS | OMP_MAP_IMPLICIT = 533
// CK26-DAG: [[TYPES:@.+]] = {{.+}}constant [3 x i64] [i64 531, i64 531, i64 531]

float Vector[1024];
#pragma omp declare target link(Vector)

extern int a;
#pragma omp declare target link(a)

double *ptr;
#pragma omp declare target link(ptr)

void declare_target_link()
{
#pragma omp target defaultmap(none:scalar) defaultmap(none:aggregate) defaultmap(none:pointer)
  {

    // CK26-DAG: call i32 @__tgt_target_mapper(i64 {{.+}}, i8* {{.+}}, i32 3, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZES]]{{.+}}, {{.+}}[[TYPES]]{{.+}}, i8** null)
    // CK26-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
    // CK26-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
    // CK26-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 2
    // CK26-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 2
    // CK26-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to double****
    // CK26-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to double***
    // CK26-DAG: store double*** @ptr_decl_tgt_ref_ptr, double**** [[CBP1]]
    // CK26-DAG: store double** @ptr, double*** [[CP1]]

    Vector[a]++;
    ptr++;
  }
}

#endif
#endif
