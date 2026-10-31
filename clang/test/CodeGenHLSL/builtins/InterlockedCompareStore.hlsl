// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-library %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,DXCHECK

// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -triple \
// RUN:   spirv-pc-vulkan-library %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,SPVCHECK

// Test basic lowering of HLSL InterlockedCompareStore to `cmpxchg monotonic`.
// The operation reports nothing, so it has a single 3-argument form and the
// `cmpxchg` result stays unused. Each test tracks the two value arguments from
// their parameters to the `cmpxchg`, so a swap of compare and value fails.

groupshared int  gs_i32;
groupshared uint gs_u32;
groupshared int64_t  gs_i64;
groupshared uint64_t gs_u64;

// CHECK-LABEL: define {{.*}}void @{{.*}}test_int
// CHECK: [[CMP:%.*]] = load i32, ptr %cmp.addr
// CHECK: [[VAL:%.*]] = load i32, ptr %v.addr
// CHECK: cmpxchg ptr addrspace(3) @gs_i32, i32 [[CMP]], i32 [[VAL]] syncscope("workgroup") monotonic monotonic
export void test_int(int cmp, int v) {
  InterlockedCompareStore(gs_i32, cmp, v);
}

// CHECK-LABEL: define {{.*}}void @{{.*}}test_uint
// CHECK: [[CMP:%.*]] = load i32, ptr %cmp.addr
// CHECK: [[VAL:%.*]] = load i32, ptr %v.addr
// CHECK: cmpxchg ptr addrspace(3) @gs_u32, i32 [[CMP]], i32 [[VAL]] syncscope("workgroup") monotonic monotonic
export void test_uint(uint cmp, uint v) {
  InterlockedCompareStore(gs_u32, cmp, v);
}

// CHECK-LABEL: define {{.*}}void @{{.*}}test_int64
// CHECK: [[CMP:%.*]] = load i64, ptr %cmp.addr
// CHECK: [[VAL:%.*]] = load i64, ptr %v.addr
// CHECK: cmpxchg ptr addrspace(3) @gs_i64, i64 [[CMP]], i64 [[VAL]] syncscope("workgroup") monotonic monotonic
export void test_int64(int64_t cmp, int64_t v) {
  InterlockedCompareStore(gs_i64, cmp, v);
}

// CHECK-LABEL: define {{.*}}void @{{.*}}test_uint64
// CHECK: [[CMP:%.*]] = load i64, ptr %cmp.addr
// CHECK: [[VAL:%.*]] = load i64, ptr %v.addr
// CHECK: cmpxchg ptr addrspace(3) @gs_u64, i64 [[CMP]], i64 [[VAL]] syncscope("workgroup") monotonic monotonic
export void test_uint64(uint64_t cmp, uint64_t v) {
  InterlockedCompareStore(gs_u64, cmp, v);
}

// A device-address-space destination uses the "device" scope instead.
RWBuffer<uint> Buf : register(u0);

// CHECK-LABEL: define {{.*}}void @{{.*}}test_device
// CHECK: [[CMP:%.*]] = load i32, ptr %cmp.addr
// CHECK: [[VAL:%.*]] = load i32, ptr %v.addr
// DXCHECK:  cmpxchg ptr %{{.*}}, i32 [[CMP]], i32 [[VAL]] syncscope("device") monotonic monotonic
// SPVCHECK: cmpxchg ptr addrspace(11) %{{.*}}, i32 [[CMP]], i32 [[VAL]] syncscope("device") monotonic monotonic
export void test_device(uint cmp, uint v) {
  InterlockedCompareStore(Buf[0], cmp, v);
}
