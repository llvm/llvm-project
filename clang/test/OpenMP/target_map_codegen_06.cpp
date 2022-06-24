// expected-no-diagnostics
#ifndef HEADER
#define HEADER

///==========================================================================///
// RUN: %clang_cc1 -no-opaque-pointers -DCK7 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK7 --check-prefix CK7-64
// RUN: %clang_cc1 -no-opaque-pointers -DCK7 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK7  --check-prefix CK7-64
// RUN: %clang_cc1 -no-opaque-pointers -DCK7 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK7  --check-prefix CK7-32
// RUN: %clang_cc1 -no-opaque-pointers -DCK7 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK7  --check-prefix CK7-32

// RUN: %clang_cc1 -no-opaque-pointers -DCK7 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK7 --check-prefix CK7-64
// RUN: %clang_cc1 -no-opaque-pointers -DCK7 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK7  --check-prefix CK7-64
// RUN: %clang_cc1 -no-opaque-pointers -DCK7 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK7  --check-prefix CK7-32
// RUN: %clang_cc1 -no-opaque-pointers -DCK7 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK7  --check-prefix CK7-32

// RUN: %clang_cc1 -no-opaque-pointers -DCK7 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK7 --check-prefix CK7-64
// RUN: %clang_cc1 -no-opaque-pointers -DCK7 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK7  --check-prefix CK7-64
// RUN: %clang_cc1 -no-opaque-pointers -DCK7 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK7  --check-prefix CK7-32
// RUN: %clang_cc1 -no-opaque-pointers -DCK7 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK7  --check-prefix CK7-32

// RUN: %clang_cc1 -no-opaque-pointers -DCK7 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY6 %s
// RUN: %clang_cc1 -no-opaque-pointers -DCK7 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY6 %s
// RUN: %clang_cc1 -no-opaque-pointers -DCK7 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY6 %s
// RUN: %clang_cc1 -no-opaque-pointers -DCK7 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY6 %s
// SIMD-ONLY6-NOT: {{__kmpc|__tgt}}
#ifdef CK7

// For a 32-bit targets, the value doesn't fit the size of the pointer,
// therefore it is passed by reference with a map 'to' specification.

// CK7-LABEL: @.__omp_offloading_{{.*}}implicit_maps_double{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0

// CK7-DAG: [[SIZES:@.+]] = {{.+}}constant [1 x i64] [i64 8]
// Map types: OMP_MAP_PRIVATE_VAL | OMP_MAP_TARGET_PARAM | OMP_MAP_IMPLICIT = 800
// CK7-64-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i64] [i64 800]
// Map types: OMP_MAP_TO  | OMP_MAP_PRIVATE | OMP_MAP_TARGET_PARAM | OMP_MAP_IMPLICIT = 673
// CK7-32-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i64] [i64 673]

// CK7-LABEL: implicit_maps_double{{.*}}(
void implicit_maps_double (int a){
  double d = (double)a;

// CK7-DAG: call i32 @__tgt_target_kernel(%struct.ident_t* @{{.+}}, i64 -1, i32 -1, i32 0, i8* @.{{.+}}.region_id, %struct.__tgt_kernel_arguments* [[ARGS:%.+]])
// CK7-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK7-DAG: store i8** [[BPGEP:%.+]], i8*** [[BPARG]]
// CK7-DAG: [[PGEP:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK7-DAG: store i8** [[PGEP:%.+]], i8*** [[BPARG]]
// CK7-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
// CK7-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
// CK7-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
// CK7-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0

// CK7-64-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to i[[sz:64|32]]*
// CK7-64-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i[[sz]]*
// CK7-64-DAG: store i[[sz]] [[VAL:%[^,]+]], i[[sz]]* [[CBP1]]
// CK7-64-DAG: store i[[sz]] [[VAL]], i[[sz]]* [[CP1]]
// CK7-64-DAG: [[VAL]] = load i[[sz]], i[[sz]]* [[ADDR:%.+]],
// CK7-64-64-DAG: [[CADDR:%.+]] = bitcast i[[sz]]* [[ADDR]] to double*
// CK7-64-64-DAG: store double {{.+}}, double* [[CADDR]],

// CK7-32-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to double**
// CK7-32-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to double**
// CK7-32-DAG: store double* [[DECL:%[^,]+]], double** [[CBP1]]
// CK7-32-DAG: store double* [[DECL]], double** [[CP1]]

// CK7-64: call void [[KERNEL:@.+]](i[[sz]] [[VAL]])
// CK7-32: call void [[KERNEL:@.+]](double* [[DECL]])
#pragma omp target
  {
    d += 1.0;
  }
}

// CK7-64: define internal void [[KERNEL]](i[[sz]] noundef [[ARG:%.+]])
// CK7-64: [[ADDR:%.+]] = alloca i[[sz]],
// CK7-64: store i[[sz]] [[ARG]], i[[sz]]* [[ADDR]],
// CK7-64: [[CADDR:%.+]] = bitcast i64* [[ADDR]] to double*
// CK7-64: {{.+}} = load double, double* [[CADDR]],

// CK7-32: define internal void [[KERNEL]](double* {{.+}}[[ARG:%.+]])
// CK7-32: [[ADDR:%.+]] = alloca double*,
// CK7-32: store double* [[ARG]], double** [[ADDR]],
// CK7-32: [[REF:%.+]] = load double*, double** [[ADDR]],
// CK7-32: {{.+}} = load double, double* [[REF]],

#endif // CK7
#endif
