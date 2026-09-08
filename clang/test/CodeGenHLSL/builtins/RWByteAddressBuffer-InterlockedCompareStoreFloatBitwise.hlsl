// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-library %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,DXCHECK

// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   spirv-pc-vulkan1.3-library %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,SPVCHECK

// Test that the RWByteAddressBuffer::InterlockedCompareStoreFloatBitwise
// member method lowers to `resource_getpointer -> cmpxchg`, for both DXIL and
// SPIR-V targets. The float arguments become their bit patterns first, because
// `cmpxchg` takes an integer.

RWByteAddressBuffer BAB : register(u0);

// CHECK-LABEL: define {{.*}}void @{{.*}}test_bab_float
// DXCHECK:  %[[HANDLE:.*]] = load target("dx.RawBuffer", i8, 1, 0), ptr {{.*}}
// DXCHECK:  %[[PTR:.*]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i8_1_0t.i32(target("dx.RawBuffer", i8, 1, 0) %[[HANDLE]], i32 %{{.*}})
// SPVCHECK: %[[HANDLE:.*]] = load target("spirv.VulkanBuffer", [0 x i8], 12, 1), ptr {{.*}}
// SPVCHECK: %[[PTR:.*]] = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i8_12_1t.i32(target("spirv.VulkanBuffer", [0 x i8], 12, 1) %[[HANDLE]], i32 %{{.*}})
// CHECK: %[[CMP:.*]] = bitcast float %{{.*}} to i32
// CHECK-NEXT: %[[VAL:.*]] = bitcast float %{{.*}} to i32
// DXCHECK-NEXT:  cmpxchg ptr %[[PTR]], i32 %[[CMP]], i32 %[[VAL]] syncscope("device") monotonic monotonic
// SPVCHECK-NEXT: cmpxchg ptr addrspace(11) %[[PTR]], i32 %[[CMP]], i32 %[[VAL]] syncscope("device") monotonic monotonic
export void test_bab_float(uint off, float cmp, float v) {
  BAB.InterlockedCompareStoreFloatBitwise(off, cmp, v);
}
