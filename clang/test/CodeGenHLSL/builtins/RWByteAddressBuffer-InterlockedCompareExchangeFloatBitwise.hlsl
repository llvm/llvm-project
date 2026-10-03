// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-library %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,DXCHECK

// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   spirv-pc-vulkan1.3-library %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,SPVCHECK

// Test that the RWByteAddressBuffer::InterlockedCompareExchangeFloatBitwise
// member method lowers to `resource_getpointer -> cmpxchg`, for both DXIL and
// SPIR-V targets. The float arguments become their bit patterns first, because
// `cmpxchg` takes an integer. The method takes the reported value by
// reference, so the pointer is loaded before the store.

RWByteAddressBuffer BAB : register(u0);

// CHECK-LABEL: define {{.*}}void @{{.*}}test_bab_float
// DXCHECK:  %[[HANDLE:.*]] = load target("dx.RawBuffer", i8, 1, 0), ptr {{.*}}
// DXCHECK:  %[[PTR:.*]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i8_1_0t.i32(target("dx.RawBuffer", i8, 1, 0) %[[HANDLE]], i32 %{{.*}})
// SPVCHECK: %[[HANDLE:.*]] = load target("spirv.VulkanBuffer", [0 x i8], 12, 1), ptr {{.*}}
// SPVCHECK: %[[PTR:.*]] = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i8_12_1t.i32(target("spirv.VulkanBuffer", [0 x i8], 12, 1) %[[HANDLE]], i32 %{{.*}})
// CHECK: %[[CMP:.*]] = bitcast float %{{.*}} to i32
// CHECK-NEXT: %[[VAL:.*]] = bitcast float %{{.*}} to i32
// DXCHECK-NEXT:  %[[PAIR:.*]] = cmpxchg ptr %[[PTR]], i32 %[[CMP]], i32 %[[VAL]] syncscope("device") monotonic monotonic
// SPVCHECK-NEXT: %[[PAIR:.*]] = cmpxchg ptr addrspace(11) %[[PTR]], i32 %[[CMP]], i32 %[[VAL]] syncscope("device") monotonic monotonic
// CHECK-NEXT: %[[RES:.*]] = extractvalue { i32, i1 } %[[PAIR]], 0
// CHECK-NEXT: %[[ORIG:.*]] = bitcast i32 %[[RES]] to float
// CHECK-NEXT: %[[DEST:.*]] = load ptr, ptr %OriginalValue.addr
// CHECK-NEXT: store float %[[ORIG]], ptr %[[DEST]]
export void test_bab_float(uint off, float cmp, float v, out float orig) {
  BAB.InterlockedCompareExchangeFloatBitwise(off, cmp, v, orig);
}
