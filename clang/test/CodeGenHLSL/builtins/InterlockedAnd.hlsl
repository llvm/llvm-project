// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-library %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,DXCHECK

// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -triple \
// RUN:   spirv-pc-vulkan-library %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,SPVCHECK

// Test basic lowering of HLSL InterlockedAnd to `atomicrmw and monotonic`.

groupshared int  gs_i32;
groupshared uint gs_u32;
groupshared int64_t  gs_i64;
groupshared uint64_t gs_u64;

// CHECK-LABEL: define {{(dso_local |hidden |internal |protected |spir_func )*}}void @{{.*}}test_int_2arg
// DXCHECK:  atomicrmw and ptr addrspace(3) {{.*}}@gs_i32{{.*}}, i32 %{{.*}} syncscope("workgroup") monotonic
// SPVCHECK: atomicrmw and ptr addrspace(3) {{.*}}@gs_i32{{.*}}, i32 %{{.*}} syncscope("workgroup") monotonic
export void test_int_2arg(int v) {
  InterlockedAnd(gs_i32, v);
}

// CHECK-LABEL: define {{(dso_local |hidden |internal |protected |spir_func )*}}void @{{.*}}test_uint_2arg
// DXCHECK:  atomicrmw and ptr addrspace(3) {{.*}}@gs_u32{{.*}}, i32 %{{.*}} syncscope("workgroup") monotonic
// SPVCHECK: atomicrmw and ptr addrspace(3) {{.*}}@gs_u32{{.*}}, i32 %{{.*}} syncscope("workgroup") monotonic
export void test_uint_2arg(uint v) {
  InterlockedAnd(gs_u32, v);
}

// CHECK-LABEL: define {{(dso_local |hidden |internal |protected |spir_func )*}}void @{{.*}}test_int_3arg
// DXCHECK:  %[[R:.*]] = atomicrmw and ptr addrspace(3) {{.*}}@gs_i32{{.*}}, i32 %{{.*}} syncscope("workgroup") monotonic
// SPVCHECK: %[[R:.*]] = atomicrmw and ptr addrspace(3) {{.*}}@gs_i32{{.*}}, i32 %{{.*}} syncscope("workgroup") monotonic
// CHECK:    store i32 %[[R]], ptr {{.*}}
export void test_int_3arg(int v, out int orig) {
  InterlockedAnd(gs_i32, v, orig);
}

// CHECK-LABEL: define {{(dso_local |hidden |internal |protected |spir_func )*}}void @{{.*}}test_uint_3arg
// DXCHECK:  %[[R:.*]] = atomicrmw and ptr addrspace(3) {{.*}}@gs_u32{{.*}}, i32 %{{.*}} syncscope("workgroup") monotonic
// SPVCHECK: %[[R:.*]] = atomicrmw and ptr addrspace(3) {{.*}}@gs_u32{{.*}}, i32 %{{.*}} syncscope("workgroup") monotonic
// CHECK:    store i32 %[[R]], ptr {{.*}}
export void test_uint_3arg(uint v, out uint orig) {
  InterlockedAnd(gs_u32, v, orig);
}

// CHECK-LABEL: define {{(dso_local |hidden |internal |protected |spir_func )*}}void @{{.*}}test_int64_2arg
// DXCHECK:  atomicrmw and ptr addrspace(3) {{.*}}@gs_i64{{.*}}, i64 %{{.*}} syncscope("workgroup") monotonic
// SPVCHECK: atomicrmw and ptr addrspace(3) {{.*}}@gs_i64{{.*}}, i64 %{{.*}} syncscope("workgroup") monotonic
export void test_int64_2arg(int64_t v) {
  InterlockedAnd(gs_i64, v);
}

// CHECK-LABEL: define {{(dso_local |hidden |internal |protected |spir_func )*}}void @{{.*}}test_uint64_3arg
// DXCHECK:  %[[R:.*]] = atomicrmw and ptr addrspace(3) {{.*}}@gs_u64{{.*}}, i64 %{{.*}} syncscope("workgroup") monotonic
// SPVCHECK: %[[R:.*]] = atomicrmw and ptr addrspace(3) {{.*}}@gs_u64{{.*}}, i64 %{{.*}} syncscope("workgroup") monotonic
// CHECK:    store i64 %[[R]], ptr {{.*}}
export void test_uint64_3arg(uint64_t v, out uint64_t orig) {
  InterlockedAnd(gs_u64, v, orig);
}
