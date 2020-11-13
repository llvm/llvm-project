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

// CK1: [[MTYPE00:@.+]] = {{.*}}constant [1 x i64] [i64 99]
// CK1: [[MTYPE01:@.+]] = {{.*}}constant [1 x i64] [i64 288]
// CK1: [[MTYPE02:@.+]] = {{.*}}constant [1 x i64] [i64 288]

void add_one(float *b, int dm)
{
  // CK1:     [[BP:%.+]] = getelementptr inbounds [1 x i8*], [1 x i8*]* %{{.+}}, i32 0, i32 0
  // CK1:     [[CBP:%.+]] = bitcast i8** [[BP]] to float**
  // CK1:     store float* [[B_ADDR:%.+]], float** [[CBP]]
  // CK1:     call void @__tgt_target_data_begin{{.+}}[[MTYPE00]]
  // CK1:     [[VAL:%.+]] = load float*, float** [[CBP]],
  // CK1-NOT: store float* [[VAL]], float** [[DECL]],
  // CK1:     store float* [[VAL]], float** [[PVT:%.+]],
  // CK1:     [[TT:%.+]] = load float*, float** [[PVT]],
  // CK1:     call i32 @__tgt_target{{.+}}[[MTYPE01]]
  // CK1:     call i32 @__tgt_target{{.+}}[[MTYPE02]]
  // CK1:     call void @__tgt_target_data_end{{.+}}[[MTYPE00]]
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
