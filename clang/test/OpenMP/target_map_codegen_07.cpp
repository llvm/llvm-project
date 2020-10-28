// expected-no-diagnostics
#ifndef HEADER
#define HEADER

///==========================================================================///
// RUN: %clang_cc1 -DCK8 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK8
// RUN: %clang_cc1 -DCK8 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK8
// RUN: %clang_cc1 -DCK8 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK8
// RUN: %clang_cc1 -DCK8 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK8

// RUN: %clang_cc1 -DCK8 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK8
// RUN: %clang_cc1 -DCK8 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK8
// RUN: %clang_cc1 -DCK8 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK8
// RUN: %clang_cc1 -DCK8 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK8

// RUN: %clang_cc1 -DCK8 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK8
// RUN: %clang_cc1 -DCK8 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK8
// RUN: %clang_cc1 -DCK8 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK8
// RUN: %clang_cc1 -DCK8 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK8

// RUN: %clang_cc1 -DCK8 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY7 %s
// RUN: %clang_cc1 -DCK8 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY7 %s
// RUN: %clang_cc1 -DCK8 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY7 %s
// RUN: %clang_cc1 -DCK8 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY7 %s
// SIMD-ONLY7-NOT: {{__kmpc|__tgt}}
#ifdef CK8

// CK8-LABEL: @.__omp_offloading_{{.*}}implicit_maps_float{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0

// CK8-DAG: [[SIZES:@.+]] = {{.+}}constant [1 x i64] [i64 4]
// Map types: OMP_MAP_PRIVATE_VAL | OMP_MAP_TARGET_PARAM | OMP_MAP_IMPLICIT = 800
// CK8-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i64] [i64 800]

// CK8-LABEL: implicit_maps_float{{.*}}(
void implicit_maps_float (int a){
  float f = (float)a;

  // CK8-DAG: call i32 @__tgt_target_mapper(i64 {{.+}}, i8* {{.+}}, i32 1, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZES]]{{.+}}, {{.+}}[[TYPES]]{{.+}}, i8** null)
  // CK8-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK8-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK8-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK8-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK8-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to i[[sz:64|32]]*
  // CK8-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i[[sz]]*
  // CK8-DAG: store i[[sz]] [[VAL:%[^,]+]], i[[sz]]* [[CBP1]]
  // CK8-DAG: store i[[sz]] [[VAL]], i[[sz]]* [[CP1]]
  // CK8-DAG: [[VAL]] = load i[[sz]], i[[sz]]* [[ADDR:%.+]],
  // CK8-DAG: [[CADDR:%.+]] = bitcast i[[sz]]* [[ADDR]] to float*
  // CK8-DAG: store float {{.+}}, float* [[CADDR]],

  // CK8: call void [[KERNEL:@.+]](i[[sz]] [[VAL]])
  #pragma omp target
  {
    f += 1.0;
  }
}

// CK8: define internal void [[KERNEL]](i[[sz]] [[ARG:%.+]])
// CK8: [[ADDR:%.+]] = alloca i[[sz]],
// CK8: store i[[sz]] [[ARG]], i[[sz]]* [[ADDR]],
// CK8: [[CADDR:%.+]] = bitcast i[[sz]]* [[ADDR]] to float*
// CK8: {{.+}} = load float, float* [[CADDR]],

#endif // CK8
#endif
