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

// Base class with an empty-for-layout member: return coercion must include the
// non-empty base subobject as a nested coerce struct (not only direct fields).
struct BaseWithEmptyMember {
  int a[2];
  OnlyStatic sub;
  float b;
  __device__ static int c;
};
struct DerivedWithBase : BaseWithEmptyMember {
  double d;
};
__device__ int BaseWithEmptyMember::c = 12;

__device__ DerivedWithBase returnDerivedWithBase() {
  OnlyStatic::something = 12;
  BaseWithEmptyMember::c = 42;
  DerivedWithBase r{};
  r.a[0] = 8;
  r.a[1] = 16;
  r.b = 3.14f;
  r.d = 1.60218e-19;
  return r;
}

__device__ void callerDerived() {
  DerivedWithBase r = returnDerivedWithBase();
  (void)r.b;
}

// CHECK-DAG: define dso_local %[[TY:.*]] @_Z16returnWithStaticv
// CHECK-DAG: call %[[TY]] @_Z16returnWithStaticv
// CHECK-DAG: %[[TY]] = type { [2 x i32], %struct.OnlyStatic, float, double }

// CHECK-DAG: define dso_local %[[DER:.*]] @_Z21returnDerivedWithBasev
// CHECK-DAG: call %[[DER]] @_Z21returnDerivedWithBasev
// CHECK-DAG: %[[BASE:[0-9]+]] = type { [2 x i32], %struct.OnlyStatic, float }
// CHECK-DAG: %[[DER]] = type { %[[BASE]], double }
