// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-library %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,DXCHECK

// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   spirv-pc-vulkan1.3-library %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,SPVCHECK

// Test that the signed and unsigned RWByteAddressBuffer::InterlockedMax and
// InterlockedMax64 methods lower through a resource pointer.

RWByteAddressBuffer BAB : register(u0);

// CHECK-LABEL: define {{.*}}void @{{.*}}test_bab_int
// DXCHECK:  %[[PTR:.*]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i8_1_0t.i32({{.*}})
// DXCHECK:  atomicrmw max ptr %[[PTR]], i32 %{{.*}} syncscope("device") monotonic
// SPVCHECK: %[[PTR:.*]] = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i8_12_1t.i32({{.*}})
// SPVCHECK: atomicrmw max ptr addrspace(11) %[[PTR]], i32 %{{.*}} syncscope("device") monotonic
export void test_bab_int(uint off, int v) {
  BAB.InterlockedMax(off, v);
}

// CHECK-LABEL: define {{.*}}void @{{.*}}test_bab_uint
// DXCHECK:  %[[PTR:.*]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i8_1_0t.i32({{.*}})
// DXCHECK:  atomicrmw umax ptr %[[PTR]], i32 %{{.*}} syncscope("device") monotonic
// SPVCHECK: %[[PTR:.*]] = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i8_12_1t.i32({{.*}})
// SPVCHECK: atomicrmw umax ptr addrspace(11) %[[PTR]], i32 %{{.*}} syncscope("device") monotonic
export void test_bab_uint(uint off, uint v) {
  BAB.InterlockedMax(off, v);
}

// CHECK-LABEL: define {{.*}}void @{{.*}}test_bab_int_orig
// DXCHECK:  %[[PTR:.*]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i8_1_0t.i32({{.*}})
// DXCHECK:  %[[R:.*]] = atomicrmw max ptr %[[PTR]], i32 %{{.*}} syncscope("device") monotonic
// DXCHECK:  store i32 %[[R]], ptr {{.*}}
// SPVCHECK: %[[PTR:.*]] = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i8_12_1t.i32({{.*}})
// SPVCHECK: %[[R:.*]] = atomicrmw max ptr addrspace(11) %[[PTR]], i32 %{{.*}} syncscope("device") monotonic
// SPVCHECK: store i32 %[[R]], ptr {{.*}}
export void test_bab_int_orig(uint off, int v, out int orig) {
  BAB.InterlockedMax(off, v, orig);
}

// CHECK-LABEL: define {{.*}}void @{{.*}}test_bab_uint_orig
// DXCHECK:  %[[PTR:.*]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i8_1_0t.i32({{.*}})
// DXCHECK:  %[[R:.*]] = atomicrmw umax ptr %[[PTR]], i32 %{{.*}} syncscope("device") monotonic
// DXCHECK:  store i32 %[[R]], ptr {{.*}}
// SPVCHECK: %[[PTR:.*]] = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i8_12_1t.i32({{.*}})
// SPVCHECK: %[[R:.*]] = atomicrmw umax ptr addrspace(11) %[[PTR]], i32 %{{.*}} syncscope("device") monotonic
// SPVCHECK: store i32 %[[R]], ptr {{.*}}
export void test_bab_uint_orig(uint off, uint v, out uint orig) {
  BAB.InterlockedMax(off, v, orig);
}

// CHECK-LABEL: define {{.*}}void @{{.*}}test_bab_int64
// DXCHECK:  %[[PTR:.*]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i8_1_0t.i32({{.*}})
// DXCHECK:  atomicrmw max ptr %[[PTR]], i64 %{{.*}} syncscope("device") monotonic
// SPVCHECK: %[[PTR:.*]] = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i8_12_1t.i32({{.*}})
// SPVCHECK: atomicrmw max ptr addrspace(11) %[[PTR]], i64 %{{.*}} syncscope("device") monotonic
export void test_bab_int64(uint off, int64_t v) {
  BAB.InterlockedMax64(off, v);
}

// CHECK-LABEL: define {{.*}}void @{{.*}}test_bab_uint64
// DXCHECK:  %[[PTR:.*]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i8_1_0t.i32({{.*}})
// DXCHECK:  atomicrmw umax ptr %[[PTR]], i64 %{{.*}} syncscope("device") monotonic
// SPVCHECK: %[[PTR:.*]] = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i8_12_1t.i32({{.*}})
// SPVCHECK: atomicrmw umax ptr addrspace(11) %[[PTR]], i64 %{{.*}} syncscope("device") monotonic
export void test_bab_uint64(uint off, uint64_t v) {
  BAB.InterlockedMax64(off, v);
}

// CHECK-LABEL: define {{.*}}void @{{.*}}test_bab_int64_orig
// DXCHECK:  %[[PTR:.*]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i8_1_0t.i32({{.*}})
// DXCHECK:  %[[R:.*]] = atomicrmw max ptr %[[PTR]], i64 %{{.*}} syncscope("device") monotonic
// DXCHECK:  store i64 %[[R]], ptr {{.*}}
// SPVCHECK: %[[PTR:.*]] = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i8_12_1t.i32({{.*}})
// SPVCHECK: %[[R:.*]] = atomicrmw max ptr addrspace(11) %[[PTR]], i64 %{{.*}} syncscope("device") monotonic
// SPVCHECK: store i64 %[[R]], ptr {{.*}}
export void test_bab_int64_orig(uint off, int64_t v, out int64_t orig) {
  BAB.InterlockedMax64(off, v, orig);
}

// CHECK-LABEL: define {{.*}}void @{{.*}}test_bab_uint64_orig
// DXCHECK:  %[[PTR:.*]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i8_1_0t.i32({{.*}})
// DXCHECK:  %[[R:.*]] = atomicrmw umax ptr %[[PTR]], i64 %{{.*}} syncscope("device") monotonic
// DXCHECK:  store i64 %[[R]], ptr {{.*}}
// SPVCHECK: %[[PTR:.*]] = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i8_12_1t.i32({{.*}})
// SPVCHECK: %[[R:.*]] = atomicrmw umax ptr addrspace(11) %[[PTR]], i64 %{{.*}} syncscope("device") monotonic
// SPVCHECK: store i64 %[[R]], ptr {{.*}}
export void test_bab_uint64_orig(uint off, uint64_t v, out uint64_t orig) {
  BAB.InterlockedMax64(off, v, orig);
}
