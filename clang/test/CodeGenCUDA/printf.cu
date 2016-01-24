// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fcuda-is-device -emit-llvm \
// RUN:   -o - %s | FileCheck %s

#include "Inputs/cuda.h"

extern "C" __device__ int vprintf(const char*, const char*);

// Check a simple call to printf end-to-end.
__device__ int CheckSimple() {
  // CHECK: [[FMT:%[0-9]+]] = load{{.*}}%fmt
  const char* fmt = "%d";
  // CHECK: [[BUF:%[a-zA-Z0-9_]+]] = alloca i8, i32 4, align 4
  // CHECK: [[PTR:%[0-9]+]] = getelementptr i8, i8* [[BUF]], i32 0
  // CHECK: [[CAST:%[0-9]+]] = bitcast i8* [[PTR]] to i32*
  // CHECK: store i32 42, i32* [[CAST]], align 4
  // CHECK: [[RET:%[0-9]+]] = call i32 @vprintf(i8* [[FMT]], i8* [[BUF]])
  // CHECK: ret i32 [[RET]]
  return printf(fmt, 42);
}

// Check that the args' types are promoted correctly when we call printf.
__device__ void CheckTypes() {
  // CHECK: alloca {{.*}} align 8
  // CHECK: getelementptr {{.*}} i32 0
  // CHECK: bitcast {{.*}} to i32*
  // CHECK: getelementptr {{.*}} i32 4
  // CHECK: bitcast {{.*}} to i32*
  // CHECK: getelementptr {{.*}} i32 8
  // CHECK: bitcast {{.*}} to double*
  // CHECK: getelementptr {{.*}} i32 16
  // CHECK: bitcast {{.*}} to double*
  printf("%d %d %f %f", (char)1, (short)2, 3.0f, 4.0);
}

// Check that the args are aligned properly in the buffer.
__device__ void CheckAlign() {
  // CHECK: alloca i8, i32 40, align 8
  // CHECK: getelementptr {{.*}} i32 0
  // CHECK: getelementptr {{.*}} i32 8
  // CHECK: getelementptr {{.*}} i32 16
  // CHECK: getelementptr {{.*}} i32 20
  // CHECK: getelementptr {{.*}} i32 24
  // CHECK: getelementptr {{.*}} i32 32
  printf("%d %f %d %d %d %lld", 1, 2.0, 3, 4, 5, (long long)6);
}

__device__ void CheckNoArgs() {
  // CHECK: call i32 @vprintf({{.*}}, i8* null){{$}}
  printf("hello, world!");
}
