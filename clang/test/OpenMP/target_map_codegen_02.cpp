// expected-no-diagnostics
#ifndef HEADER
#define HEADER

///==========================================================================///
// RUN: %clang_cc1 -no-opaque-pointers -DCK3 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK3 --check-prefix CK3-64
// RUN: %clang_cc1 -no-opaque-pointers -DCK3 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK3 --check-prefix CK3-64
// RUN: %clang_cc1 -no-opaque-pointers -DCK3 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK3 --check-prefix CK3-32
// RUN: %clang_cc1 -no-opaque-pointers -DCK3 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK3 --check-prefix CK3-32

// RUN: %clang_cc1 -no-opaque-pointers -DCK3 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK3 --check-prefix CK3-64
// RUN: %clang_cc1 -no-opaque-pointers -DCK3 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK3 --check-prefix CK3-64
// RUN: %clang_cc1 -no-opaque-pointers -DCK3 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK3 --check-prefix CK3-32
// RUN: %clang_cc1 -no-opaque-pointers -DCK3 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK3 --check-prefix CK3-32

// RUN: %clang_cc1 -no-opaque-pointers -DCK3 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK3 --check-prefix CK3-64
// RUN: %clang_cc1 -no-opaque-pointers -DCK3 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK3 --check-prefix CK3-64
// RUN: %clang_cc1 -no-opaque-pointers -DCK3 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK3 --check-prefix CK3-32
// RUN: %clang_cc1 -no-opaque-pointers -DCK3 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK3 --check-prefix CK3-32

// RUN: %clang_cc1 -no-opaque-pointers -DCK3 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY2 %s
// RUN: %clang_cc1 -no-opaque-pointers -DCK3 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY2 %s
// RUN: %clang_cc1 -no-opaque-pointers -DCK3 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY2 %s
// RUN: %clang_cc1 -no-opaque-pointers -DCK3 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY2 %s
// SIMD-ONLY2-NOT: {{__kmpc|__tgt}}
#ifdef CK3

// CK3-LABEL: @.__omp_offloading_{{.*}}implicit_maps_parameter{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0

// CK3-DAG: [[SIZES:@.+]] = {{.+}}constant [1 x i64] [i64 4]
// Map types: OMP_MAP_PRIVATE_VAL | OMP_MAP_TARGET_PARAM | OMP_MAP_IMPLICIT = 800
// CK3-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i64] [i64 800]

// CK3-LABEL: implicit_maps_parameter{{.*}}(
void implicit_maps_parameter (int a){

// CK3-DAG: call i32 @__tgt_target_kernel(%struct.ident_t* @{{.+}}, i64 -1, i32 -1, i32 0, i8* @.{{.+}}.region_id, %struct.__tgt_kernel_arguments* [[ARGS:%.+]])
// CK3-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK3-DAG: store i8** [[BPGEP:%.+]], i8*** [[BPARG]]
// CK3-DAG: [[PGEP:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK3-DAG: store i8** [[PGEP:%.+]], i8*** [[BPARG]]
// CK3-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
// CK3-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
// CK3-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
// CK3-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
// CK3-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to i[[sz:64|32]]*
// CK3-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i[[sz]]*
// CK3-DAG: store i[[sz]] [[VAL:%[^,]+]], i[[sz]]* [[CBP1]]
// CK3-DAG: store i[[sz]] [[VAL]], i[[sz]]* [[CP1]]
// CK3-DAG: [[VAL]] = load i[[sz]], i[[sz]]* [[ADDR:%.+]],
// CK3-64-DAG: [[CADDR:%.+]] = bitcast i[[sz]]* [[ADDR]] to i32*
// CK3-64-DAG: store i32 {{.+}}, i32* [[CADDR]],

// CK3: call void [[KERNEL:@.+]](i[[sz]] [[VAL]])
#pragma omp target
  {
   ++a;
  }
}

// CK3: define internal void [[KERNEL]](i[[sz]] noundef [[ARG:%.+]])
// CK3: [[ADDR:%.+]] = alloca i[[sz]],
// CK3: store i[[sz]] [[ARG]], i[[sz]]* [[ADDR]],
// CK3-64: [[CADDR:%.+]] = bitcast i64* [[ADDR]] to i32*
// CK3-64: {{.+}} = load i32, i32* [[CADDR]],
// CK3-32: {{.+}} = load i32, i32* [[ADDR]],

#endif // CK3
#endif
