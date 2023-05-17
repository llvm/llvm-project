// expected-no-diagnostics
#ifndef HEADER
#define HEADER



///==========================================================================///
// RUN: %clang_cc1 -DCK2 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK2 --check-prefix CK2-64
// RUN: %clang_cc1 -DCK2 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK2 --check-prefix CK2-64
// RUN: %clang_cc1 -DCK2 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK2 --check-prefix CK2-32
// RUN: %clang_cc1 -DCK2 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK2 --check-prefix CK2-32

// RUN: %clang_cc1 -DCK2 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK2 --check-prefix CK2-64
// RUN: %clang_cc1 -DCK2 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK2 --check-prefix CK2-64
// RUN: %clang_cc1 -DCK2 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK2 --check-prefix CK2-32
// RUN: %clang_cc1 -DCK2 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK2 --check-prefix CK2-32

// RUN: %clang_cc1 -DCK2 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK2 --check-prefix CK2-64
// RUN: %clang_cc1 -DCK2 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK2 --check-prefix CK2-64
// RUN: %clang_cc1 -DCK2 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK2 --check-prefix CK2-32
// RUN: %clang_cc1 -DCK2 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK2 --check-prefix CK2-32

// RUN: %clang_cc1 -DCK2 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -DCK2 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -DCK2 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -DCK2 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY1 %s
// SIMD-ONLY1-NOT: {{__kmpc|__tgt}}
#ifdef CK2

// CK2-LABEL: @.__omp_offloading_{{.*}}implicit_maps_reference{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0

// CK2: [[SIZES:@.+]] = {{.+}}constant [1 x i64] [i64 4]
// Map types: OMP_MAP_PRIVATE_VAL | OMP_MAP_TARGET_PARAM | OMP_MAP_IMPLICIT = 800
// CK2: [[TYPES:@.+]] = {{.+}}constant [1 x i64] [i64 800]
// CK2-LABEL: @.__omp_offloading_{{.*}}implicit_maps_reference{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK2: [[SIZES2:@.+]] = {{.+}}constant [1 x i64] zeroinitializer
// Map types: OMP_MAP_IS_PTR | OMP_MAP_IMPLICIT = 544
// CK2: [[TYPES2:@.+]] = {{.+}}constant [1 x i64] [i64 544]

// CK2-LABEL: implicit_maps_reference{{.*}}(
void implicit_maps_reference (int a, int *b){
  int &i = a;
// CK2-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK2-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK2-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK2-DAG: [[PGEP:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK2-DAG: store ptr [[PGEP:%.+]], ptr [[BPARG]]
// CK2-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
// CK2-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
// CK2-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
// CK2-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
// CK2-DAG: store i[[sz:64|32]] [[VAL:%[^,]+]], ptr [[BP1]]
// CK2-DAG: store i[[sz]] [[VAL]], ptr [[P1]]
// CK2-DAG: [[VAL]] = load i[[sz]], ptr [[ADDR:%.+]],
// CK2-64-DAG: store i32 {{.+}}, ptr [[ADDR]],

// CK2: call void [[KERNEL:@.+]](i[[sz]] [[VAL]])
#pragma omp target
  {
   ++i;
  }

  int *&p = b;
// CK2-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK2-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK2-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK2-DAG: [[PGEP:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK2-DAG: store ptr [[PGEP:%.+]], ptr [[BPARG]]
// CK2-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
// CK2-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
// CK2-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
// CK2-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
// CK2-DAG: store ptr [[VAL:%[^,]+]], ptr [[BP1]]
// CK2-DAG: store ptr [[VAL]], ptr [[P1]]
// CK2-DAG: [[VAL]] = load ptr, ptr [[ADDR:%.+]],
// CK2-DAG: [[ADDR]] = load ptr, ptr [[ADDR2:%.+]],

// CK2: call void [[KERNEL2:@.+]](ptr [[VAL]])
#pragma omp target
  {
   ++p;
  }
}

// CK2: define internal void [[KERNEL]](i[[sz]] noundef [[ARG:%.+]])
// CK2: [[ADDR:%.+]] = alloca i[[sz]],
// CK2: [[REF:%.+]] = alloca ptr,
// CK2: store i[[sz]] [[ARG]], ptr [[ADDR]],
// CK2-64: store ptr [[ADDR]], ptr [[REF]],
// CK2-64: [[RVAL:%.+]] = load ptr, ptr [[REF]],
// CK2-64: {{.+}} = load i32, ptr [[RVAL]],
// CK2-32: store ptr [[ADDR]], ptr [[REF]],
// CK2-32: [[RVAL:%.+]] = load ptr, ptr [[REF]],
// CK2-32: {{.+}} = load i32, ptr [[RVAL]],

// CK2: define internal void [[KERNEL2]](ptr noundef [[ARG:%.+]])
// CK2: [[ADDR:%.+]] = alloca ptr,
// CK2: [[REF:%.+]] = alloca ptr,
// CK2: store ptr [[ARG]], ptr [[ADDR]],
// CK2: store ptr [[ADDR]], ptr [[REF]],
// CK2: [[T:%.+]] = load ptr, ptr [[REF]],
// CK2: [[TT:%.+]] = load ptr, ptr [[T]],
// CK2: getelementptr inbounds i32, ptr [[TT]], i32 1
#endif // CK2
#endif
