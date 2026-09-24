// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-library %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,DXCHECK

// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   spirv-pc-vulkan1.3-library %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,SPVCHECK

// Test that the RWByteAddressBuffer::InterlockedExchange and
// InterlockedExchange64 member methods lower to `resource_getpointer ->
// atomicrmw xchg`, and that the returned original value is stored through the
// out parameter, for both DXIL and SPIR-V targets.

RWByteAddressBuffer BAB : register(u0);

// CHECK-LABEL: define {{.*}}void @{{.*}}test_bab_uint_3arg
// DXCHECK:  %[[HANDLE:.*]] = load target("dx.RawBuffer", i8, 1, 0), ptr {{.*}}
// DXCHECK:  %[[PTR:.*]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i8_1_0t.i32(target("dx.RawBuffer", i8, 1, 0) %[[HANDLE]], i32 %{{.*}})
// DXCHECK:  %[[R:.*]] = atomicrmw xchg ptr %[[PTR]], i32 %{{.*}} syncscope("device") monotonic
// DXCHECK:  store i32 %[[R]], ptr {{.*}}
// SPVCHECK: %[[HANDLE:.*]] = load target("spirv.VulkanBuffer", [0 x i8], 12, 1), ptr {{.*}}
// SPVCHECK: %[[PTR:.*]] = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i8_12_1t.i32(target("spirv.VulkanBuffer", [0 x i8], 12, 1) %[[HANDLE]], i32 %{{.*}})
// SPVCHECK: %[[R:.*]] = atomicrmw xchg ptr addrspace(11) %[[PTR]], i32 %{{.*}} syncscope("device") monotonic
// SPVCHECK: store i32 %[[R]], ptr {{.*}}
export void test_bab_uint_3arg(uint off, uint v, out uint orig) {
  BAB.InterlockedExchange(off, v, orig);
}

// CHECK-LABEL: define {{.*}}void @{{.*}}test_bab_uint64_3arg
// DXCHECK:  %[[HANDLE:.*]] = load target("dx.RawBuffer", i8, 1, 0), ptr {{.*}}
// DXCHECK:  %[[PTR:.*]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i8_1_0t.i32(target("dx.RawBuffer", i8, 1, 0) %[[HANDLE]], i32 %{{.*}})
// DXCHECK:  %[[R:.*]] = atomicrmw xchg ptr %[[PTR]], i64 %{{.*}} syncscope("device") monotonic
// DXCHECK:  store i64 %[[R]], ptr {{.*}}
// SPVCHECK: %[[HANDLE:.*]] = load target("spirv.VulkanBuffer", [0 x i8], 12, 1), ptr {{.*}}
// SPVCHECK: %[[PTR:.*]] = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i8_12_1t.i32(target("spirv.VulkanBuffer", [0 x i8], 12, 1) %[[HANDLE]], i32 %{{.*}})
// SPVCHECK: %[[R:.*]] = atomicrmw xchg ptr addrspace(11) %[[PTR]], i64 %{{.*}} syncscope("device") monotonic
// SPVCHECK: store i64 %[[R]], ptr {{.*}}
export void test_bab_uint64_3arg(uint off, uint64_t v, out uint64_t orig) {
  BAB.InterlockedExchange64(off, v, orig);
}
