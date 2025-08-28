// REQUIRES: asserts
// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clang -fcxx-exceptions -Xclang -disable-llvm-passes -S -emit-llvm -Wripple -fenable-ripple -o - %s | FileCheck --implicit-check-not="warning:" %s

#include <ripple.h>

// CHECK-LABEL @test1
extern "C" void test1(size_t n, float * C, float * A, float * B) {
  ripple_block_t BS = ripple_set_block_shape(0, 8);

// CHECK: for.body:
// CHECK: call void @__cxa_throw
// CHECK: ripple.par.for.remainder.body:                    ; preds = %ripple.par.for.remainder.cond
// CHECK: call void @__cxa_throw
  ripple_parallel(BS, 1);
  for (size_t j = 0; j <= n; j++) {
    ripple_parallel(BS, 0);
    for (size_t i = 0; i <= n; i++) {
      if (B[i] == 0.f)
        throw "Divide by zero!";
      C[i] = A[i] / B[i];
    }
  }
}

// CHECK-LABEL: @test2
extern "C" const char* test2(size_t n, float * C, float * A, float * B) {
  ripple_block_t BS = ripple_set_block_shape(0, 8);

// CHECK: for.body:
// CHECK: invoke void @__cxa_throw
// CHECK-NEXT: to label{{.*}}unwind label %[[LPAD1:[0-9a-zA-Z._]+]]
// CHECK: ripple.par.for.remainder.body:                    ; preds = %ripple.par.for.remainder.cond
// CHECK: invoke void @__cxa_throw
// CHECK-NOT: to label{{.*}}unwind label %[[LPAD1]]{{$}}
  ripple_parallel(BS, 1);
  for (size_t j = 0; j <= n; j++) {
    ripple_parallel(BS, 0);
    for (size_t i = 0; i <= n; i++) {
      try {
        if (B[i] == 0.f)
          throw "Divide by zero!";
        C[i] = A[i] / B[i];
      } catch (const char* msg) {
        return msg;
      }
    }
  }
  return "";
}

// CHECK-LABEL: @test3
extern "C" const char* test3(size_t n, float * C, float * A, float * B) {
  ripple_block_t BS = ripple_set_block_shape(0, 8, 8);

  try {
// CHECK: for.body:
// CHECK: invoke void @__cxa_throw
// CHECK-NEXT: to label{{.*}}unwind label %[[LPAD1:[0-9a-zA-Z._]+]]
// CHECK: ripple.par.for.remainder.body:                    ; preds = %ripple.par.for.remainder.cond
// CHECK: invoke void @__cxa_throw
// CHECK-NEXT: to label{{.*}}unwind label %[[LPAD1]]{{$}}
    ripple_parallel(BS, 1);
    for (size_t j = 0; j <= n; j++) {
      ripple_parallel(BS, 0);
      for (size_t i = 0; i <= n; i++) {
        if (B[i] == 0.f)
          throw "Divide by zero!";
        C[i] = A[i] / B[i];
      }
    }
  } catch (const char* msg) {
    return msg;
  }
  return "";
}
