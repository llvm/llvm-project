// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -triple \
// RUN:   spirv1.6-unknown-vulkan1.3-compute %s -S -o - | FileCheck %s

// An Interlocked op on a groupshared destination must use the Workgroup scope
// (2), not CrossDevice (which the backend emits as OpConstantNull).

groupshared uint gs;

// CHECK-DAG: %[[#UINT:]] = OpTypeInt 32 0
// CHECK-DAG: %[[#WORKGROUP:]] = OpConstant %[[#UINT]] 2
// CHECK: OpAtomicOr %[[#UINT]] %[[#]] %[[#WORKGROUP]] %[[#]] %[[#]]
// CHECK-NOT: OpConstantNull

[numthreads(1,1,1)]
void main() {
  InterlockedOr(gs, 1);
}
