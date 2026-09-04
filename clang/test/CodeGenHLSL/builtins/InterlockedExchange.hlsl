// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-library %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,DXCHECK

// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -triple \
// RUN:   spirv-pc-vulkan-library %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,SPVCHECK

// Test basic lowering of HLSL InterlockedExchange to `atomicrmw xchg
// monotonic`. InterlockedExchange always reports the previous value, so it
// only has a 3-argument form.

groupshared int  gs_i32;
groupshared uint gs_u32;
groupshared int64_t  gs_i64;
groupshared uint64_t gs_u64;

// CHECK-LABEL: define {{.*}}void @{{.*}}test_int_3arg
// DXCHECK:  %[[R:.*]] = atomicrmw xchg ptr addrspace(3) {{.*}}@gs_i32{{.*}}, i32 %{{.*}} syncscope("workgroup") monotonic
// SPVCHECK: %[[R:.*]] = atomicrmw xchg ptr addrspace(3) {{.*}}@gs_i32{{.*}}, i32 %{{.*}} syncscope("workgroup") monotonic
// CHECK:    store i32 %[[R]], ptr {{.*}}
export void test_int_3arg(int v, out int orig) {
  InterlockedExchange(gs_i32, v, orig);
}

// CHECK-LABEL: define {{.*}}void @{{.*}}test_uint_3arg
// DXCHECK:  %[[R:.*]] = atomicrmw xchg ptr addrspace(3) {{.*}}@gs_u32{{.*}}, i32 %{{.*}} syncscope("workgroup") monotonic
// SPVCHECK: %[[R:.*]] = atomicrmw xchg ptr addrspace(3) {{.*}}@gs_u32{{.*}}, i32 %{{.*}} syncscope("workgroup") monotonic
// CHECK:    store i32 %[[R]], ptr {{.*}}
export void test_uint_3arg(uint v, out uint orig) {
  InterlockedExchange(gs_u32, v, orig);
}

// CHECK-LABEL: define {{.*}}void @{{.*}}test_int64_3arg
// DXCHECK:  %[[R:.*]] = atomicrmw xchg ptr addrspace(3) {{.*}}@gs_i64{{.*}}, i64 %{{.*}} syncscope("workgroup") monotonic
// SPVCHECK: %[[R:.*]] = atomicrmw xchg ptr addrspace(3) {{.*}}@gs_i64{{.*}}, i64 %{{.*}} syncscope("workgroup") monotonic
// CHECK:    store i64 %[[R]], ptr {{.*}}
export void test_int64_3arg(int64_t v, out int64_t orig) {
  InterlockedExchange(gs_i64, v, orig);
}

// CHECK-LABEL: define {{.*}}void @{{.*}}test_uint64_3arg
// DXCHECK:  %[[R:.*]] = atomicrmw xchg ptr addrspace(3) {{.*}}@gs_u64{{.*}}, i64 %{{.*}} syncscope("workgroup") monotonic
// SPVCHECK: %[[R:.*]] = atomicrmw xchg ptr addrspace(3) {{.*}}@gs_u64{{.*}}, i64 %{{.*}} syncscope("workgroup") monotonic
// CHECK:    store i64 %[[R]], ptr {{.*}}
export void test_uint64_3arg(uint64_t v, out uint64_t orig) {
  InterlockedExchange(gs_u64, v, orig);
}
