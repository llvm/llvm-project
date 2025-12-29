// expected-no-diagnostics
#ifndef HEADER
#define HEADER
///==========================================================================///
// RUN: %clang_cc1 -DCK1 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK1 --check-prefix CK1-64
// RUN: %clang_cc1 -DCK1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK1 --check-prefix CK1-64
// RUN: %clang_cc1 -DCK1 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK1 --check-prefix CK1-32
// RUN: %clang_cc1 -DCK1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK1 --check-prefix CK1-32

// RUN: %clang_cc1 -DCK1 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -DCK1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -DCK1 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -DCK1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// SIMD-ONLY1-NOT: {{__kmpc|__tgt}}
#ifdef CK1

// CK1: [[MYSIZE00:@.+]] = {{.*}}constant [2 x i64] [i64 4, i64 {{8|4}}]
// CK1: [[MTYPE00:@.+]] = {{.*}}constant [2 x i64] [i64 67, i64 16384]
// CK1: [[MTYPE01:@.+]] = {{.*}}constant [1 x i64] [i64 288]
// CK1: [[MTYPE02:@.+]] = {{.*}}constant [1 x i64] [i64 288]

void add_one(float *b, int dm)
{
  // &B[0], &B[0], 1 * sizeof(B[0]), PARAM | TO | FROM
  // &B,    &B[0], sizeof(B),        ATTACH

  // CK1:     [[RB_1:%.*]] = load ptr, ptr [[B:%b.addr]]
  // CK1:     [[RB_2:%.*]] = load ptr, ptr [[B]]
  // CK1:     [[RB0_1:%.*]] = getelementptr inbounds nuw float, ptr [[RB_2]], i{{.*}} 0

  // CK1:     [[BP0:%.+]] = getelementptr inbounds [2 x ptr], ptr [[BP:%.offload_baseptrs.*]], i32 0, i32 0
  // CK1:     store ptr [[RB_1]], ptr [[BP0]]
  // CK1:     [[P0:%.+]] = getelementptr inbounds [2 x ptr], ptr [[P:%.offload_ptrs.*]], i32 0, i32 0
  // CK1:     store ptr [[RB0_1]], ptr [[P0]]

  // CK1:     [[BP1:%.+]] = getelementptr inbounds [2 x ptr], ptr [[BP]], i32 0, i32 1
  // CK1:     store ptr [[B]], ptr [[BP1]]
  // CK1:     [[P1:%.+]] = getelementptr inbounds [2 x ptr], ptr [[P]], i32 0, i32 1
  // CK1:     store ptr [[RB0_1]], ptr [[P1]]

  // CK1:     call void @__tgt_target_data_begin{{.+}}ptr [[MYSIZE00]], ptr [[MTYPE00]]

  // CK1:     [[VAL:%.+]] = load ptr, ptr [[BP0]],
  // CK1-NOT: store ptr [[VAL]], ptr {{%.+}},
  // CK1:     store ptr [[VAL]], ptr [[PVT:%.+]],
  // CK1:     [[TT:%.+]] = load ptr, ptr [[PVT]],
  // CK1:     call i32 @__tgt_target{{.+}}
  // CK1:     call i32 @__tgt_target{{.+}}
  // CK1:     call void @__tgt_target_data_end{{.+}}ptr [[MYSIZE00]], ptr [[MTYPE00]]
#pragma omp target data map(tofrom:b[:1]) use_device_ptr(b) if(dm == 0)
  {
#pragma omp target is_device_ptr(b)
  {
    b[0] += 1;
  }
  }
}

#endif
#endif
