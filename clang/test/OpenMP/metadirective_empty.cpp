// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-unknown-linux -emit-llvm %s -fexceptions -fcxx-exceptions -o - | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple aarch64-unknown-linux -emit-llvm %s -fexceptions -fcxx-exceptions -o - | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple ppc64le-unknown-linux -emit-llvm %s -fexceptions -fcxx-exceptions -o - | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

#define N 1000
void func() {
  // Test where a valid when clause contains empty directive.
  // The directive will be ignored and code for a serial for loop will be generated.
#pragma omp metadirective when(implementation = {vendor(llvm)} \
                               :) default(parallel for)
  for (int i = 0; i < N; i++)
    ;
}

// CHECK-LABEL: void @_Z4funcv()
// CHECK: entry:
// CHECK:   [[I:%.+]] = alloca i32,
// CHECK:   store i32 0, ptr [[I]],
// CHECK:   br label %[[FOR_COND:.+]]
// CHECK: [[FOR_COND]]:
// CHECK:   [[ZERO:%.+]] = load i32, ptr [[I]],
// CHECK:   [[CMP:%.+]] = icmp slt i32 [[ZERO]], 1000
// CHECK:   br i1 [[CMP]], label %[[FOR_BODY:.+]], label %[[FOR_END:.+]]
// CHECK: [[FOR_BODY]]:
// CHECK:   br label %[[FOR_INC:.+]]
// CHECK: [[FOR_INC]]:
// CHECK:   [[ONE:%.+]] = load i32, ptr [[I]],
// CHECK:   [[INC:%.+]] = add nsw i32 [[ONE]], 1
// CHECK:   store i32 [[INC]], ptr [[I]],
// CHECK:   br label %[[FOR_COND]],
// CHECK: [[FOR_END]]:
// CHECK:   ret void
// CHECK: }

#endif
