// expected-no-diagnostics
#ifndef HEADER
#define HEADER

///==========================================================================///

// RUN: %clang_cc1 -DCK11 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefixes CK11,CK11_4
// RUN: %clang_cc1 -DCK11 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes CK11,CK11_4
// RUN: %clang_cc1 -DCK11 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes CK11,CK11_4
// RUN: %clang_cc1 -DCK11 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes CK11,CK11_4

// RUN: %clang_cc1 -DCK11 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap %s --check-prefixes CK11,CK11_5
// RUN: %clang_cc1 -DCK11 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes CK11,CK11_5
// RUN: %clang_cc1 -DCK11 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes CK11,CK11_5
// RUN: %clang_cc1 -DCK11 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes CK11,CK11_5

// RUN: %clang_cc1 -DCK11 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY10 %s
// RUN: %clang_cc1 -DCK11 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY10 %s
// RUN: %clang_cc1 -DCK11 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY10 %s
// RUN: %clang_cc1 -DCK11 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY10 %s
// SIMD-ONLY10-NOT: {{__kmpc|__tgt}}
#ifdef CK11

// CK11-LABEL: @.__omp_offloading_{{.*}}implicit_maps_double_complex{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK11_4-DAG: [[SIZES:@.+]] = {{.+}}constant [2 x i64] [i64 16, i64 {{8|4}}]
// CK11_5-DAG: [[SIZES:@.+]] = {{.+}}constant [2 x i64] [i64 16, i64 0]
// Map types: OMP_MAP_TO  | OMP_MAP_FROM | OMP_MAP_TARGET_PARAM | OMP_MAP_IMPLICIT = 547
// CK11_4-DAG: [[TYPES:@.+]] = {{.+}}constant [2 x i64] [i64 547, i64 547]
// CK11_5-DAG: [[TYPES:@.+]] = {{.+}}constant [2 x i64] [i64 547, i64 544]

// CK11-LABEL: implicit_maps_double_complex{{.*}}(
void implicit_maps_double_complex (int a, int *b){
  double _Complex dc = (double)a;

// CK11-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK11-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK11-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK11-DAG: [[PGEP:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK11-DAG: store ptr [[PGEP:%.+]], ptr [[BPARG]]
// CK11-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
// CK11-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
// CK11-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
// CK11-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
// CK11-DAG: store ptr [[PTR:%[^,]+]], ptr [[BP1]]
// CK11-DAG: store ptr [[PTR]], ptr [[P1]]


// CK11: call void [[KERNEL:@.+]](ptr [[PTR]], ptr %{{.+}})
#pragma omp target defaultmap(tofrom \
                              : scalar)
  {
   dc *= dc; *b = 1;
  }
}

// CK11: define internal void [[KERNEL]](ptr {{.*}}[[ARG:%.+]], ptr {{.*}})
// CK11: [[ADDR:%.+]] = alloca ptr,
// CK11: store ptr [[ARG]], ptr [[ADDR]],
// CK11: [[REF:%.+]] = load ptr, ptr [[ADDR]],
// CK11: {{.+}} = getelementptr inbounds nuw { double, double }, ptr [[REF]], i32 0, i32 0
#endif // CK11
#endif
