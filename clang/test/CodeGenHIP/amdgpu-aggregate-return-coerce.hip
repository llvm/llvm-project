// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -emit-llvm -fcuda-is-device -o - %s \
// RUN:   | FileCheck %s

#define __device__ __attribute__((device))

struct OnlyStatic {
  __device__ static int something;
};
struct WithStaticFields {
  int a[2];
  OnlyStatic sub;
  float b;
  __device__ static int c;
  double d;
};
__device__ int OnlyStatic::something = 42;
__device__ int WithStaticFields::c = 12;

__device__ WithStaticFields returnWithStatic() {
  OnlyStatic::something = 12;
  WithStaticFields::c = 42;
  return {.a = {8, 16}, .b = 3.14f, .d = 1.60218e-19};
}

__device__ void caller() {
  WithStaticFields r = returnWithStatic();
  (void)r.b;
}

// CHECK-DAG: define dso_local %[[TY:.*]] @_Z16returnWithStaticv
// CHECK-DAG: call %[[TY]] @_Z16returnWithStaticv
// CHECK-DAG: %[[TY]] = type { [2 x i32], %struct.OnlyStatic, float, double }
