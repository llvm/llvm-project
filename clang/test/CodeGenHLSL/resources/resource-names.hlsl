// REQUIRES: spirv-registered-target
// RUN: %clang_cc1 -triple spirv1.6-unknown-vulkan1.3-compute -S -O1 -o - %s | FileCheck %s

// Resource names must not introduce byte arrays or require Int8 support.
// CHECK-NOT: OpCapability Int8
// CHECK: OpName [[BUF:%[0-9]+]] "buf"
// CHECK-NOT: OpTypeInt 8
// CHECK: [[BUF]] = OpVariable {{.*}} StorageBuffer

RWStructuredBuffer<int> buf : register(u0);

[numthreads(1, 1, 1)]
void main() {
  buf[0] = 1;
}
