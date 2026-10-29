// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-library %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,DXCHECK

// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   spirv-pc-vulkan1.3-library %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,SPVCHECK

// Test that the RWByteAddressBuffer::InterlockedCompareStore and
// InterlockedCompareStore64 member methods lower to `resource_getpointer ->
// cmpxchg`, for both DXIL and SPIR-V targets. Compare-store reports nothing,
// so there is no out parameter and the `cmpxchg` result stays unused.

RWByteAddressBuffer BAB : register(u0);

// CHECK-LABEL: define {{.*}}void @{{.*}}test_bab_uint
// DXCHECK:  %[[HANDLE:.*]] = load target("dx.RawBuffer", i8, 1, 0), ptr {{.*}}
// DXCHECK:  %[[PTR:.*]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i8_1_0t.i32(target("dx.RawBuffer", i8, 1, 0) %[[HANDLE]], i32 %{{.*}})
// DXCHECK:  cmpxchg ptr %[[PTR]], i32 %{{.*}}, i32 %{{.*}} syncscope("device") monotonic monotonic
// SPVCHECK: %[[HANDLE:.*]] = load target("spirv.VulkanBuffer", [0 x i8], 12, 1), ptr {{.*}}
// SPVCHECK: %[[PTR:.*]] = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i8_12_1t.i32(target("spirv.VulkanBuffer", [0 x i8], 12, 1) %[[HANDLE]], i32 %{{.*}})
// SPVCHECK: cmpxchg ptr addrspace(11) %[[PTR]], i32 %{{.*}}, i32 %{{.*}} syncscope("device") monotonic monotonic
export void test_bab_uint(uint off, uint cmp, uint v) {
  BAB.InterlockedCompareStore(off, cmp, v);
}

// CHECK-LABEL: define {{.*}}void @{{.*}}test_bab_uint64
// DXCHECK:  %[[HANDLE:.*]] = load target("dx.RawBuffer", i8, 1, 0), ptr {{.*}}
// DXCHECK:  %[[PTR:.*]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i8_1_0t.i32(target("dx.RawBuffer", i8, 1, 0) %[[HANDLE]], i32 %{{.*}})
// DXCHECK:  cmpxchg ptr %[[PTR]], i64 %{{.*}}, i64 %{{.*}} syncscope("device") monotonic monotonic
// SPVCHECK: %[[HANDLE:.*]] = load target("spirv.VulkanBuffer", [0 x i8], 12, 1), ptr {{.*}}
// SPVCHECK: %[[PTR:.*]] = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i8_12_1t.i32(target("spirv.VulkanBuffer", [0 x i8], 12, 1) %[[HANDLE]], i32 %{{.*}})
// SPVCHECK: cmpxchg ptr addrspace(11) %[[PTR]], i64 %{{.*}}, i64 %{{.*}} syncscope("device") monotonic monotonic
export void test_bab_uint64(uint off, uint64_t cmp, uint64_t v) {
  BAB.InterlockedCompareStore64(off, cmp, v);
}
