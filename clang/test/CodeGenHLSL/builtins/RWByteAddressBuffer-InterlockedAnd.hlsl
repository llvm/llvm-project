// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-library %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,DXCHECK

// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   spirv-pc-vulkan1.3-library %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,SPVCHECK

// Test that the RWByteAddressBuffer::InterlockedAnd and InterlockedAnd64
// member methods lower to `resource_getpointer -> atomicrmw and`, and that
// the 3-argument overload stores the returned original value through the
// out parameter, for both DXIL and SPIR-V targets.

RWByteAddressBuffer BAB : register(u0);

// CHECK-LABEL: define {{(dso_local |hidden |internal |protected |spir_func )*}}void @{{.*}}test_bab_int_2arg
// DXCHECK:  %[[HANDLE:.*]] = load target("dx.RawBuffer", i8, 1, 0), ptr {{.*}}
// DXCHECK:  %[[PTR:.*]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i8_1_0t.i32(target("dx.RawBuffer", i8, 1, 0) %[[HANDLE]], i32 %{{.*}})
// DXCHECK:  atomicrmw and ptr %[[PTR]], i32 %{{.*}} syncscope("device") monotonic
// SPVCHECK: %[[HANDLE:.*]] = load target("spirv.VulkanBuffer", [0 x i8], 12, 1), ptr {{.*}}
// SPVCHECK: %[[PTR:.*]] = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i8_12_1t.i32(target("spirv.VulkanBuffer", [0 x i8], 12, 1) %[[HANDLE]], i32 %{{.*}})
// SPVCHECK: atomicrmw and ptr addrspace(11) %[[PTR]], i32 %{{.*}} syncscope("device") monotonic
export void test_bab_int_2arg(uint off, int v) {
  BAB.InterlockedAnd(off, v);
}

// CHECK-LABEL: define {{(dso_local |hidden |internal |protected |spir_func )*}}void @{{.*}}test_bab_uint_3arg
// DXCHECK:  %[[HANDLE:.*]] = load target("dx.RawBuffer", i8, 1, 0), ptr {{.*}}
// DXCHECK:  %[[PTR:.*]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i8_1_0t.i32(target("dx.RawBuffer", i8, 1, 0) %[[HANDLE]], i32 %{{.*}})
// DXCHECK:  %[[R:.*]] = atomicrmw and ptr %[[PTR]], i32 %{{.*}} syncscope("device") monotonic
// DXCHECK:  store i32 %[[R]], ptr {{.*}}
// SPVCHECK: %[[HANDLE:.*]] = load target("spirv.VulkanBuffer", [0 x i8], 12, 1), ptr {{.*}}
// SPVCHECK: %[[PTR:.*]] = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i8_12_1t.i32(target("spirv.VulkanBuffer", [0 x i8], 12, 1) %[[HANDLE]], i32 %{{.*}})
// SPVCHECK: %[[R:.*]] = atomicrmw and ptr addrspace(11) %[[PTR]], i32 %{{.*}} syncscope("device") monotonic
// SPVCHECK: store i32 %[[R]], ptr {{.*}}
export void test_bab_uint_3arg(uint off, uint v, out uint orig) {
  BAB.InterlockedAnd(off, v, orig);
}

// CHECK-LABEL: define {{(dso_local |hidden |internal |protected |spir_func )*}}void @{{.*}}test_bab_int64_2arg
// DXCHECK:  %[[HANDLE:.*]] = load target("dx.RawBuffer", i8, 1, 0), ptr {{.*}}
// DXCHECK:  %[[PTR:.*]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i8_1_0t.i32(target("dx.RawBuffer", i8, 1, 0) %[[HANDLE]], i32 %{{.*}})
// DXCHECK:  atomicrmw and ptr %[[PTR]], i64 %{{.*}} syncscope("device") monotonic
// SPVCHECK: %[[HANDLE:.*]] = load target("spirv.VulkanBuffer", [0 x i8], 12, 1), ptr {{.*}}
// SPVCHECK: %[[PTR:.*]] = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i8_12_1t.i32(target("spirv.VulkanBuffer", [0 x i8], 12, 1) %[[HANDLE]], i32 %{{.*}})
// SPVCHECK: atomicrmw and ptr addrspace(11) %[[PTR]], i64 %{{.*}} syncscope("device") monotonic
export void test_bab_int64_2arg(uint off, int64_t v) {
  BAB.InterlockedAnd64(off, v);
}

// CHECK-LABEL: define {{(dso_local |hidden |internal |protected |spir_func )*}}void @{{.*}}test_bab_uint64_3arg
// DXCHECK:  %[[HANDLE:.*]] = load target("dx.RawBuffer", i8, 1, 0), ptr {{.*}}
// DXCHECK:  %[[PTR:.*]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i8_1_0t.i32(target("dx.RawBuffer", i8, 1, 0) %[[HANDLE]], i32 %{{.*}})
// DXCHECK:  %[[R:.*]] = atomicrmw and ptr %[[PTR]], i64 %{{.*}} syncscope("device") monotonic
// DXCHECK:  store i64 %[[R]], ptr {{.*}}
// SPVCHECK: %[[HANDLE:.*]] = load target("spirv.VulkanBuffer", [0 x i8], 12, 1), ptr {{.*}}
// SPVCHECK: %[[PTR:.*]] = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i8_12_1t.i32(target("spirv.VulkanBuffer", [0 x i8], 12, 1) %[[HANDLE]], i32 %{{.*}})
// SPVCHECK: %[[R:.*]] = atomicrmw and ptr addrspace(11) %[[PTR]], i64 %{{.*}} syncscope("device") monotonic
// SPVCHECK: store i64 %[[R]], ptr {{.*}}
export void test_bab_uint64_3arg(uint off, uint64_t v, out uint64_t orig) {
  BAB.InterlockedAnd64(off, v, orig);
}
