// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-library %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,DXCHECK

// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -triple \
// RUN:   spirv-pc-vulkan-library %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,SPVCHECK

// Test basic lowering of HLSL InterlockedCompareStoreFloatBitwise to `cmpxchg
// monotonic`. `cmpxchg` takes an integer, so the float arguments become their
// bit patterns first. The operation reports nothing, so the `cmpxchg` result
// stays unused.

groupshared float gs_f32;

// CHECK-LABEL: define {{.*}}void @{{.*}}test_float
// CHECK: [[CMP:%.*]] = bitcast float %{{.*}} to i32
// CHECK-NEXT: [[VAL:%.*]] = bitcast float %{{.*}} to i32
// DXCHECK-NEXT:  cmpxchg ptr addrspace(3) {{.*}}@gs_f32{{.*}}, i32 [[CMP]], i32 [[VAL]] syncscope("workgroup") monotonic monotonic
// SPVCHECK-NEXT: cmpxchg ptr addrspace(3) {{.*}}@gs_f32{{.*}}, i32 [[CMP]], i32 [[VAL]] syncscope("workgroup") monotonic monotonic
export void test_float(float cmp, float v) {
  InterlockedCompareStoreFloatBitwise(gs_f32, cmp, v);
}

// A device-address-space destination uses the "device" scope instead.
RWBuffer<float> Buf : register(u0);

// CHECK-LABEL: define {{.*}}void @{{.*}}test_device
// CHECK: [[CMP:%.*]] = bitcast float %{{.*}} to i32
// CHECK-NEXT: [[VAL:%.*]] = bitcast float %{{.*}} to i32
// DXCHECK-NEXT:  cmpxchg ptr %{{.*}}, i32 [[CMP]], i32 [[VAL]] syncscope("device") monotonic monotonic
// SPVCHECK-NEXT: cmpxchg ptr addrspace(11) %{{.*}}, i32 [[CMP]], i32 [[VAL]] syncscope("device") monotonic monotonic
export void test_device(float cmp, float v) {
  InterlockedCompareStoreFloatBitwise(Buf[0], cmp, v);
}
