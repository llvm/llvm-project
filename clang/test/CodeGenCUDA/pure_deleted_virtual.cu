// RUN: %clang_cc1 -emit-llvm %s -o - -fcuda-is-device -triple nvptx64 | FileCheck %s

// Check that __cxa_pure_virtual() and __cxa_deleted_virtual() are always
// available in device code.

#define __device__ __attribute__((__device__))

struct S {
  __device__ virtual void anchor();
  __device__ virtual void pure() = 0;
  __device__ virtual void deleted() = delete;
};

// Anchor function to force vtable emission.
__device__ void S::anchor() {}

// CHECK-DAG: @_ZTV1S = {{.*}} constant { [5 x ptr] } { [5 x ptr] [ptr null, ptr null, ptr @_ZN1S6anchorEv, ptr @__cxa_pure_virtual, ptr @__cxa_deleted_virtual] }
// CHECK-DAG: define weak{{.*}} void @__cxa_pure_virtual()
// CHECK-DAG: define weak{{.*}} void @__cxa_deleted_virtual()
