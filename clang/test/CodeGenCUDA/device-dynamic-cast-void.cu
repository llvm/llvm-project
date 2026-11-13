// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fcuda-is-device -emit-llvm \
// RUN:     -o - %s | FileCheck %s --implicit-check-not=_ZTI \
// RUN:     --implicit-check-not=__dynamic_cast
// RUN: %clang_cc1 -x hip -triple amdgcn-amd-amdhsa -fcuda-is-device \
// RUN:     -emit-llvm -o - %s | FileCheck %s --implicit-check-not=_ZTI \
// RUN:     --implicit-check-not=__dynamic_cast

// dynamic_cast<void*> is allowed in device code as it doesn't use RTTI since it
// only reads the offset-to-top from the vtable. With two bases, the
// offset-to-top is non-zero for the second base.

#include "Inputs/cuda.h"

struct A {
  __device__ virtual void f() {}
  int a;
};
struct B {
  __device__ virtual void g() {}
  int b;
};
struct C : A, B {
  __device__ void f() override {}
  __device__ void g() override {}
};

// The offset-to-top is 0 in the vtable for A-in-C and -16 in the vtable for
// B-in-C. The RTTI slots are null.
// CHECK: @_ZTV1C = {{.*}} constant
// CHECK-SAME: [4 x [[PTR:ptr( addrspace\(1\))?]]] {{\[}}[[PTR]] null, [[PTR]] null,
// CHECK-SAME: [3 x [[PTR]]] {{\[}}[[PTR]] inttoptr (i64 -16 to [[PTR]]), [[PTR]] null,

// CHECK-LABEL: define {{.*}} ptr @_Z15cast_first_baseP1A(
// CHECK: [[VTABLE:%vtable]] = load ptr, ptr [[THIS:%.*]], align 8
// CHECK-NEXT: [[SLOT:%.*]] = getelementptr inbounds i64, ptr [[VTABLE]], i64 -2
// CHECK-NEXT: [[OFFSET:%.*]] = load i64, ptr [[SLOT]], align 8
// CHECK-NEXT: getelementptr inbounds i8, ptr [[THIS]], i64 [[OFFSET]]
__device__ void *cast_first_base(A *p) {
  return dynamic_cast<void *>(p);
}

// CHECK-LABEL: define {{.*}} ptr @_Z16cast_second_baseP1B(
// CHECK: [[VTABLE:%vtable]] = load ptr, ptr [[THIS:%.*]], align 8
// CHECK-NEXT: [[SLOT:%.*]] = getelementptr inbounds i64, ptr [[VTABLE]], i64 -2
// CHECK-NEXT: [[OFFSET:%.*]] = load i64, ptr [[SLOT]], align 8
// CHECK-NEXT: getelementptr inbounds i8, ptr [[THIS]], i64 [[OFFSET]]
__device__ void *cast_second_base(B *p) {
  return dynamic_cast<void *>(p);
}

// Constructing C emits its vtable.
__device__ void use() {
  C c;
  cast_first_base(&c);
  cast_second_base(&c);
}
