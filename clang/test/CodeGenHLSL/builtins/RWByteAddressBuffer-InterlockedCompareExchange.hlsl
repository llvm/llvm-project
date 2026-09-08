// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-library %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,DXCHECK

// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   spirv-pc-vulkan1.3-library %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,SPVCHECK

// Test that the RWByteAddressBuffer::InterlockedCompareExchange and
// InterlockedCompareExchange64 member methods lower to `resource_getpointer ->
// cmpxchg`, for both DXIL and SPIR-V targets. The value that was in the
// destination before the operation goes to the out parameter.

RWByteAddressBuffer BAB : register(u0);

// CHECK-LABEL: define {{.*}}void @{{.*}}test_bab_uint
// DXCHECK:  %[[HANDLE:.*]] = load target("dx.RawBuffer", i8, 1, 0), ptr {{.*}}
// DXCHECK:  %[[PTR:.*]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i8_1_0t.i32(target("dx.RawBuffer", i8, 1, 0) %[[HANDLE]], i32 %{{.*}})
// DXCHECK:  %[[PAIR:.*]] = cmpxchg ptr %[[PTR]], i32 %{{.*}}, i32 %{{.*}} syncscope("device") monotonic monotonic
// SPVCHECK: %[[HANDLE:.*]] = load target("spirv.VulkanBuffer", [0 x i8], 12, 1), ptr {{.*}}
// SPVCHECK: %[[PTR:.*]] = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i8_12_1t.i32(target("spirv.VulkanBuffer", [0 x i8], 12, 1) %[[HANDLE]], i32 %{{.*}})
// SPVCHECK: %[[PAIR:.*]] = cmpxchg ptr addrspace(11) %[[PTR]], i32 %{{.*}}, i32 %{{.*}} syncscope("device") monotonic monotonic
// CHECK-NEXT: %[[OLD:.*]] = extractvalue { i32, i1 } %[[PAIR]], 0
// CHECK-NEXT: %[[OUT:.*]] = load ptr{{.*}}, ptr {{.*}}%OriginalValue.addr
// CHECK-NEXT: store i32 %[[OLD]], ptr{{.*}} %[[OUT]]
export void test_bab_uint(uint off, uint cmp, uint v) {
  uint orig;
  BAB.InterlockedCompareExchange(off, cmp, v, orig);
}

// CHECK-LABEL: define {{.*}}void @{{.*}}test_bab_uint64
// DXCHECK:  %[[HANDLE:.*]] = load target("dx.RawBuffer", i8, 1, 0), ptr {{.*}}
// DXCHECK:  %[[PTR:.*]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i8_1_0t.i32(target("dx.RawBuffer", i8, 1, 0) %[[HANDLE]], i32 %{{.*}})
// DXCHECK:  %[[PAIR:.*]] = cmpxchg ptr %[[PTR]], i64 %{{.*}}, i64 %{{.*}} syncscope("device") monotonic monotonic
// SPVCHECK: %[[HANDLE:.*]] = load target("spirv.VulkanBuffer", [0 x i8], 12, 1), ptr {{.*}}
// SPVCHECK: %[[PTR:.*]] = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i8_12_1t.i32(target("spirv.VulkanBuffer", [0 x i8], 12, 1) %[[HANDLE]], i32 %{{.*}})
// SPVCHECK: %[[PAIR:.*]] = cmpxchg ptr addrspace(11) %[[PTR]], i64 %{{.*}}, i64 %{{.*}} syncscope("device") monotonic monotonic
// CHECK-NEXT: %[[OLD:.*]] = extractvalue { i64, i1 } %[[PAIR]], 0
// CHECK-NEXT: %[[OUT:.*]] = load ptr{{.*}}, ptr {{.*}}%OriginalValue.addr
// CHECK-NEXT: store i64 %[[OLD]], ptr{{.*}} %[[OUT]]
export void test_bab_uint64(uint off, uint64_t cmp, uint64_t v) {
  uint64_t orig;
  BAB.InterlockedCompareExchange64(off, cmp, v, orig);
}
