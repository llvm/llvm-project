// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.8-library %s -emit-llvm \
// RUN:   -disable-llvm-passes -o - | FileCheck %s

RWBuffer<float> UAVBuffer;
RWByteAddressBuffer Bytes;

void test_barrier() {
  // CHECK: call void @llvm.dx.barrier.by.memory.type(i32 3, i32 5)
  Barrier(UAV_MEMORY | GROUP_SHARED_MEMORY, GROUP_SYNC | DEVICE_SCOPE);

  // CHECK: call void @llvm.dx.barrier.by.memory.handle.tdx.TypedBuffer
  // CHECK-SAME: (target("dx.TypedBuffer", float, 1, 0, 0) {{.*}}, i32 2)
  Barrier(UAVBuffer, GROUP_SCOPE);

  // CHECK: call void @llvm.dx.barrier.by.memory.handle.tdx.RawBuffer
  // CHECK-SAME: (target("dx.RawBuffer", i8, 1, 0) {{.*}}, i32 4)
  Barrier(Bytes, DEVICE_SCOPE);
}

// CHECK: declare void @llvm.dx.barrier.by.memory.type(i32, i32)
// CHECK-SAME: #[[ATTRS:[0-9]+]]
// CHECK: declare void @llvm.dx.barrier.by.memory.handle.tdx.TypedBuffer
// CHECK-SAME: #[[ATTRS]]
// CHECK: declare void @llvm.dx.barrier.by.memory.handle.tdx.RawBuffer
// CHECK-SAME: #[[ATTRS]]
// CHECK: attributes #[[ATTRS]] = {{.*}}convergent
