// Test host codegen.
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=50 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=50 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}

// expected-no-diagnostics
#ifndef HEADER
#define HEADER

void foo(int n) {

  // CHECK:       [[N:%.+]] = load i32, i32* [[N_ADDR:%.+]],
  // CHECK:       store i32 [[N]], i32* [[DEVICE_CAP:%.+]],
  // CHECK:       [[DEV:%.+]] = load i32, i32* [[DEVICE_CAP]],
  // CHECK:       [[DEVICE:%.+]] = sext i32 [[DEV]] to i64
  // CHECK:       [[RET:%.+]] = call i32 @__tgt_target_mapper(i64 [[DEVICE]], i8* @{{[^,]+}}, i32 0, i8** null, i8** null, i64* null, i64* null, i8** null)
  // CHECK-NEXT:  [[ERROR:%.+]] = icmp ne i32 [[RET]], 0
  // CHECK-NEXT:  br i1 [[ERROR]], label %[[FAIL:[^,]+]], label %[[END:[^,]+]]
  // CHECK:       [[FAIL]]
  // CHECK:       call void [[HVT0:@.+]]()
  // CHECK-NEXT:  br label %[[END]]
  // CHECK:       [[END]]
  #pragma omp target device(n)
  ;
  // CHECK:       [[N:%.+]] = load i32, i32* [[N_ADDR]],
  // CHECK:       store i32 [[N]], i32* [[DEVICE_CAP:%.+]],
  // CHECK:       [[DEV:%.+]] = load i32, i32* [[DEVICE_CAP]],
  // CHECK:       [[DEVICE:%.+]] = sext i32 [[DEV]] to i64
  // CHECK:       [[RET:%.+]] = call i32 @__tgt_target_mapper(i64 [[DEVICE]], i8* @{{[^,]+}}, i32 0, i8** null, i8** null, i64* null, i64* null, i8** null)
  // CHECK-NEXT:  [[ERROR:%.+]] = icmp ne i32 [[RET]], 0
  // CHECK-NEXT:  br i1 [[ERROR]], label %[[FAIL:[^,]+]], label %[[END:[^,]+]]
  // CHECK:       [[FAIL]]
  // CHECK:       call void [[HVT0:@.+]]()
  // CHECK-NEXT:  br label %[[END]]
  // CHECK:       [[END]]
  #pragma omp target device(device_num: n)
  ;
  // CHECK-NOT:   call i32 @__tgt_target_mapper(
  // CHECK:       call void @__omp_offloading_{{.+}}_l46()
  // CHECK-NOT:   call i32 @__tgt_target_mapper(
  #pragma omp target device(ancestor: n)
  ;
}

#endif
