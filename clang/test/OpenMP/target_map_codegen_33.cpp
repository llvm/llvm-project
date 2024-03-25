// expected-no-diagnostics
#ifndef HEADER
#define HEADER

///==========================================================================///
// RUN: %clang_cc1 -DCK32 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK32 --check-prefix CK32-64
// RUN: %clang_cc1 -DCK32 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK32 --check-prefix CK32-64
// RUN: %clang_cc1 -DCK32 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK32 --check-prefix CK32-32
// RUN: %clang_cc1 -DCK32 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK32 --check-prefix CK32-32

// RUN: %clang_cc1 -DCK32 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY32 %s
// RUN: %clang_cc1 -DCK32 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY32 %s
// RUN: %clang_cc1 -DCK32 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY32 %s
// RUN: %clang_cc1 -DCK32 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY32 %s
// SIMD-ONLY32-NOT: {{__kmpc|__tgt}}
#ifdef CK32

// CK32-DAG: [[MTYPE_TO:@.+]] = {{.+}}constant [1 x i64] [i64 33]
// CK32-DAG: [[MTYPE_FROM:@.+]] = {{.+}}constant [1 x i64] [i64 34]

void array_shaping(float *f, int sa) {

// CK32-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK32-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK32-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK32-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK32-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CK32-DAG: [[SARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 4
// CK32-DAG: store ptr [[SIZES:%.+]], ptr [[SARG]]
// CK32-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK32-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
// CK32-DAG: [[SIZES]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

// CK32-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK32-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK32-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0


// CK32-DAG: store ptr [[F1:%.+]], ptr [[BP0]],
// CK32-DAG: store ptr [[F2:%.+]], ptr [[P0]],
// CK32-DAG: store i64 [[SIZE:%.+]], ptr [[S0]],

// CK32-DAG: [[F1]] = load ptr, ptr [[F_ADDR:%.+]],
// CK32-DAG: [[F2]] = load ptr, ptr [[F_ADDR]],
// CK32-64-DAG: [[SIZE]] = mul nuw i64 [[SZ1:%.+]], 4
// CK32-64-DAG: [[SZ1]] = mul nuw i64 12, %{{.+}}
// CK32-32-DAG: [[SIZE]] = sext i32 [[SZ1:%.+]] to i64
// CK32-32-DAG: [[SZ1]] = mul nuw i32 [[SZ2:%.+]], 4
// CK32-32-DAG: [[SZ2]] = mul nuw i32 12, %{{.+}}
#pragma omp target map(to \
                       : ([3][sa][4])f)
  f[0] = 1;
  sa = 1;
// CK32-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK32-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK32-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK32-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK32-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CK32-DAG: [[SARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 4
// CK32-DAG: store ptr [[SIZES:%.+]], ptr [[SARG]]
// CK32-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK32-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
// CK32-DAG: [[SIZES]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

// CK32-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK32-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK32-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0


// CK32-DAG: store ptr [[F1:%.+]], ptr [[BP0]],
// CK32-DAG: store ptr [[F2:%.+]], ptr [[P0]],
// CK32-DAG: store i64 [[SIZE:%.+]], ptr [[S0]],

// CK32-DAG: [[F1]] = load ptr, ptr [[F_ADDR:%.+]],
// CK32-DAG: [[F2]] = load ptr, ptr [[F_ADDR]],
// CK32-64-DAG: [[SIZE]] = mul nuw i64 [[SZ1:%.+]], 5
// CK32-64-DAG: [[SZ1]] = mul nuw i64 4, %{{.+}}
// CK32-32-DAG: [[SIZE]] = sext i32 [[SZ1:%.+]] to i64
// CK32-32-DAG: [[SZ1]] = mul nuw i32 [[SZ2:%.+]], 5
// CK32-32-DAG: [[SZ2]] = mul nuw i32 4, %{{.+}}
#pragma omp target map(from \
                       : ([sa][5])f)
  f[0] = 1;
}

#endif // CK32
#endif
