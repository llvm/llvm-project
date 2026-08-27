// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-library %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,DXCHECK

// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   spirv-pc-vulkan1.3-library %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,SPVCHECK

// Test that the signed and unsigned RWByteAddressBuffer::InterlockedMin and
// InterlockedMin64 methods lower through a resource pointer.

RWByteAddressBuffer BAB : register(u0);

// CHECK-LABEL: define {{(dso_local |hidden |internal |protected |spir_func )*}}void @{{.*}}test_bab_int
// DXCHECK:  %[[PTR:.*]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i8_1_0t.i32({{.*}})
// DXCHECK:  atomicrmw min ptr %[[PTR]], i32 %{{.*}} syncscope("device") monotonic
// SPVCHECK: %[[PTR:.*]] = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i8_12_1t.i32({{.*}})
// SPVCHECK: atomicrmw min ptr addrspace(11) %[[PTR]], i32 %{{.*}} syncscope("device") monotonic
export void test_bab_int(uint off, int v) {
  BAB.InterlockedMin(off, v);
}

// CHECK-LABEL: define {{(dso_local |hidden |internal |protected |spir_func )*}}void @{{.*}}test_bab_uint
// DXCHECK:  %[[PTR:.*]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i8_1_0t.i32({{.*}})
// DXCHECK:  atomicrmw umin ptr %[[PTR]], i32 %{{.*}} syncscope("device") monotonic
// SPVCHECK: %[[PTR:.*]] = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i8_12_1t.i32({{.*}})
// SPVCHECK: atomicrmw umin ptr addrspace(11) %[[PTR]], i32 %{{.*}} syncscope("device") monotonic
export void test_bab_uint(uint off, uint v) {
  BAB.InterlockedMin(off, v);
}

// CHECK-LABEL: define {{(dso_local |hidden |internal |protected |spir_func )*}}void @{{.*}}test_bab_int_orig
// DXCHECK:  %[[PTR:.*]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i8_1_0t.i32({{.*}})
// DXCHECK:  %[[R:.*]] = atomicrmw min ptr %[[PTR]], i32 %{{.*}} syncscope("device") monotonic
// DXCHECK:  store i32 %[[R]], ptr {{.*}}
// SPVCHECK: %[[PTR:.*]] = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i8_12_1t.i32({{.*}})
// SPVCHECK: %[[R:.*]] = atomicrmw min ptr addrspace(11) %[[PTR]], i32 %{{.*}} syncscope("device") monotonic
// SPVCHECK: store i32 %[[R]], ptr {{.*}}
export void test_bab_int_orig(uint off, int v, out int orig) {
  BAB.InterlockedMin(off, v, orig);
}

// CHECK-LABEL: define {{(dso_local |hidden |internal |protected |spir_func )*}}void @{{.*}}test_bab_uint_orig
// DXCHECK:  %[[PTR:.*]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i8_1_0t.i32({{.*}})
// DXCHECK:  %[[R:.*]] = atomicrmw umin ptr %[[PTR]], i32 %{{.*}} syncscope("device") monotonic
// DXCHECK:  store i32 %[[R]], ptr {{.*}}
// SPVCHECK: %[[PTR:.*]] = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i8_12_1t.i32({{.*}})
// SPVCHECK: %[[R:.*]] = atomicrmw umin ptr addrspace(11) %[[PTR]], i32 %{{.*}} syncscope("device") monotonic
// SPVCHECK: store i32 %[[R]], ptr {{.*}}
export void test_bab_uint_orig(uint off, uint v, out uint orig) {
  BAB.InterlockedMin(off, v, orig);
}

// CHECK-LABEL: define {{(dso_local |hidden |internal |protected |spir_func )*}}void @{{.*}}test_bab_int64
// DXCHECK:  %[[PTR:.*]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i8_1_0t.i32({{.*}})
// DXCHECK:  atomicrmw min ptr %[[PTR]], i64 %{{.*}} syncscope("device") monotonic
// SPVCHECK: %[[PTR:.*]] = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i8_12_1t.i32({{.*}})
// SPVCHECK: atomicrmw min ptr addrspace(11) %[[PTR]], i64 %{{.*}} syncscope("device") monotonic
export void test_bab_int64(uint off, int64_t v) {
  BAB.InterlockedMin64(off, v);
}

// CHECK-LABEL: define {{(dso_local |hidden |internal |protected |spir_func )*}}void @{{.*}}test_bab_uint64
// DXCHECK:  %[[PTR:.*]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i8_1_0t.i32({{.*}})
// DXCHECK:  atomicrmw umin ptr %[[PTR]], i64 %{{.*}} syncscope("device") monotonic
// SPVCHECK: %[[PTR:.*]] = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i8_12_1t.i32({{.*}})
// SPVCHECK: atomicrmw umin ptr addrspace(11) %[[PTR]], i64 %{{.*}} syncscope("device") monotonic
export void test_bab_uint64(uint off, uint64_t v) {
  BAB.InterlockedMin64(off, v);
}

// CHECK-LABEL: define {{(dso_local |hidden |internal |protected |spir_func )*}}void @{{.*}}test_bab_int64_orig
// DXCHECK:  %[[PTR:.*]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i8_1_0t.i32({{.*}})
// DXCHECK:  %[[R:.*]] = atomicrmw min ptr %[[PTR]], i64 %{{.*}} syncscope("device") monotonic
// DXCHECK:  store i64 %[[R]], ptr {{.*}}
// SPVCHECK: %[[PTR:.*]] = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i8_12_1t.i32({{.*}})
// SPVCHECK: %[[R:.*]] = atomicrmw min ptr addrspace(11) %[[PTR]], i64 %{{.*}} syncscope("device") monotonic
// SPVCHECK: store i64 %[[R]], ptr {{.*}}
export void test_bab_int64_orig(uint off, int64_t v, out int64_t orig) {
  BAB.InterlockedMin64(off, v, orig);
}

// CHECK-LABEL: define {{(dso_local |hidden |internal |protected |spir_func )*}}void @{{.*}}test_bab_uint64_orig
// DXCHECK:  %[[PTR:.*]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i8_1_0t.i32({{.*}})
// DXCHECK:  %[[R:.*]] = atomicrmw umin ptr %[[PTR]], i64 %{{.*}} syncscope("device") monotonic
// DXCHECK:  store i64 %[[R]], ptr {{.*}}
// SPVCHECK: %[[PTR:.*]] = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i8_12_1t.i32({{.*}})
// SPVCHECK: %[[R:.*]] = atomicrmw umin ptr addrspace(11) %[[PTR]], i64 %{{.*}} syncscope("device") monotonic
// SPVCHECK: store i64 %[[R]], ptr {{.*}}
export void test_bab_uint64_orig(uint off, uint64_t v, out uint64_t orig) {
  BAB.InterlockedMin64(off, v, orig);
}
