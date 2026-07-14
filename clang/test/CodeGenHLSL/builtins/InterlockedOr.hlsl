// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,DXCHECK

// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -triple \
// RUN:   spirv-pc-vulkan-compute %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,SPVCHECK

// Test basic lowering of HLSL InterlockedOr to the target intrinsic.

groupshared int  gs_i32;
groupshared uint gs_u32;
groupshared int64_t  gs_i64;
groupshared uint64_t gs_u64;

// CHECK-LABEL: define {{(dso_local |hidden |internal |protected |spir_func )*}}void @{{.*}}test_int_2arg
// DXCHECK:  call i32 @llvm.dx.interlocked.or.i32.p3(ptr addrspace(3) {{.*}}@gs_i32{{.*}}, i32 %{{.*}})
// SPVCHECK: call spir_func i32 @llvm.spv.interlocked.or.i32.p3(ptr addrspace(3) {{.*}}@gs_i32{{.*}}, i32 %{{.*}})
export void test_int_2arg(int v) {
  InterlockedOr(gs_i32, v);
}

// CHECK-LABEL: define {{(dso_local |hidden |internal |protected |spir_func )*}}void @{{.*}}test_uint_2arg
// DXCHECK:  call i32 @llvm.dx.interlocked.or.i32.p3(ptr addrspace(3) {{.*}}@gs_u32{{.*}}, i32 %{{.*}})
// SPVCHECK: call spir_func i32 @llvm.spv.interlocked.or.i32.p3(ptr addrspace(3) {{.*}}@gs_u32{{.*}}, i32 %{{.*}})
export void test_uint_2arg(uint v) {
  InterlockedOr(gs_u32, v);
}

// CHECK-LABEL: define {{(dso_local |hidden |internal |protected |spir_func )*}}void @{{.*}}test_int_3arg
// DXCHECK:  %[[R:.*]] = call i32 @llvm.dx.interlocked.or.i32.p3(ptr addrspace(3) {{.*}}@gs_i32{{.*}}, i32 %{{.*}})
// SPVCHECK: %[[R:.*]] = call spir_func i32 @llvm.spv.interlocked.or.i32.p3(ptr addrspace(3) {{.*}}@gs_i32{{.*}}, i32 %{{.*}})
// CHECK:    store i32 %[[R]], ptr {{.*}}
export void test_int_3arg(int v, out int orig) {
  InterlockedOr(gs_i32, v, orig);
}

// CHECK-LABEL: define {{(dso_local |hidden |internal |protected |spir_func )*}}void @{{.*}}test_uint_3arg
// DXCHECK:  %[[R:.*]] = call i32 @llvm.dx.interlocked.or.i32.p3(ptr addrspace(3) {{.*}}@gs_u32{{.*}}, i32 %{{.*}})
// SPVCHECK: %[[R:.*]] = call spir_func i32 @llvm.spv.interlocked.or.i32.p3(ptr addrspace(3) {{.*}}@gs_u32{{.*}}, i32 %{{.*}})
// CHECK:    store i32 %[[R]], ptr {{.*}}
export void test_uint_3arg(uint v, out uint orig) {
  InterlockedOr(gs_u32, v, orig);
}

// CHECK-LABEL: define {{(dso_local |hidden |internal |protected |spir_func )*}}void @{{.*}}test_int64_2arg
// DXCHECK:  call i64 @llvm.dx.interlocked.or.i64.p3(ptr addrspace(3) {{.*}}@gs_i64{{.*}}, i64 %{{.*}})
// SPVCHECK: call spir_func i64 @llvm.spv.interlocked.or.i64.p3(ptr addrspace(3) {{.*}}@gs_i64{{.*}}, i64 %{{.*}})
export void test_int64_2arg(int64_t v) {
  InterlockedOr(gs_i64, v);
}

// CHECK-LABEL: define {{(dso_local |hidden |internal |protected |spir_func )*}}void @{{.*}}test_uint64_3arg
// DXCHECK:  %[[R:.*]] = call i64 @llvm.dx.interlocked.or.i64.p3(ptr addrspace(3) {{.*}}@gs_u64{{.*}}, i64 %{{.*}})
// SPVCHECK: %[[R:.*]] = call spir_func i64 @llvm.spv.interlocked.or.i64.p3(ptr addrspace(3) {{.*}}@gs_u64{{.*}}, i64 %{{.*}})
// CHECK:    store i64 %[[R]], ptr {{.*}}
export void test_uint64_3arg(uint64_t v, out uint64_t orig) {
  InterlockedOr(gs_u64, v, orig);
}
