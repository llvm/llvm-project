// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-library %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,DXCHECK

// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -triple \
// RUN:   spirv-pc-vulkan-library %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,SPVCHECK

// Test basic lowering of HLSL InterlockedCompareExchange to `cmpxchg
// monotonic`. The operation reports the value that was in the destination
// before the operation, so the first element of the `cmpxchg` result goes to
// the out parameter.

groupshared int  gs_i32;
groupshared uint gs_u32;
groupshared int64_t  gs_i64;
groupshared uint64_t gs_u64;

// CHECK-LABEL: define {{.*}}void @{{.*}}test_int
// DXCHECK:  [[PAIR:%.*]] = cmpxchg ptr addrspace(3) {{.*}}@gs_i32{{.*}}, i32 %{{.*}}, i32 %{{.*}} syncscope("workgroup") monotonic monotonic
// SPVCHECK: [[PAIR:%.*]] = cmpxchg ptr addrspace(3) {{.*}}@gs_i32{{.*}}, i32 %{{.*}}, i32 %{{.*}} syncscope("workgroup") monotonic monotonic
// CHECK-NEXT: [[OLD:%.*]] = extractvalue { i32, i1 } [[PAIR]], 0
// CHECK-NEXT: store i32 [[OLD]], ptr {{.*}}%orig
export void test_int(int cmp, int v) {
  int orig;
  InterlockedCompareExchange(gs_i32, cmp, v, orig);
}

// CHECK-LABEL: define {{.*}}void @{{.*}}test_uint
// DXCHECK:  [[PAIR:%.*]] = cmpxchg ptr addrspace(3) {{.*}}@gs_u32{{.*}}, i32 %{{.*}}, i32 %{{.*}} syncscope("workgroup") monotonic monotonic
// SPVCHECK: [[PAIR:%.*]] = cmpxchg ptr addrspace(3) {{.*}}@gs_u32{{.*}}, i32 %{{.*}}, i32 %{{.*}} syncscope("workgroup") monotonic monotonic
// CHECK-NEXT: [[OLD:%.*]] = extractvalue { i32, i1 } [[PAIR]], 0
// CHECK-NEXT: store i32 [[OLD]], ptr {{.*}}%orig
export void test_uint(uint cmp, uint v) {
  uint orig;
  InterlockedCompareExchange(gs_u32, cmp, v, orig);
}

// CHECK-LABEL: define {{.*}}void @{{.*}}test_int64
// DXCHECK:  [[PAIR:%.*]] = cmpxchg ptr addrspace(3) {{.*}}@gs_i64{{.*}}, i64 %{{.*}}, i64 %{{.*}} syncscope("workgroup") monotonic monotonic
// SPVCHECK: [[PAIR:%.*]] = cmpxchg ptr addrspace(3) {{.*}}@gs_i64{{.*}}, i64 %{{.*}}, i64 %{{.*}} syncscope("workgroup") monotonic monotonic
// CHECK-NEXT: [[OLD:%.*]] = extractvalue { i64, i1 } [[PAIR]], 0
// CHECK-NEXT: store i64 [[OLD]], ptr {{.*}}%orig
export void test_int64(int64_t cmp, int64_t v) {
  int64_t orig;
  InterlockedCompareExchange(gs_i64, cmp, v, orig);
}

// CHECK-LABEL: define {{.*}}void @{{.*}}test_uint64
// DXCHECK:  [[PAIR:%.*]] = cmpxchg ptr addrspace(3) {{.*}}@gs_u64{{.*}}, i64 %{{.*}}, i64 %{{.*}} syncscope("workgroup") monotonic monotonic
// SPVCHECK: [[PAIR:%.*]] = cmpxchg ptr addrspace(3) {{.*}}@gs_u64{{.*}}, i64 %{{.*}}, i64 %{{.*}} syncscope("workgroup") monotonic monotonic
// CHECK-NEXT: [[OLD:%.*]] = extractvalue { i64, i1 } [[PAIR]], 0
// CHECK-NEXT: store i64 [[OLD]], ptr {{.*}}%orig
export void test_uint64(uint64_t cmp, uint64_t v) {
  uint64_t orig;
  InterlockedCompareExchange(gs_u64, cmp, v, orig);
}

// A device-address-space destination uses the "device" scope instead.
RWBuffer<uint> Buf : register(u0);

// CHECK-LABEL: define {{.*}}void @{{.*}}test_device
// DXCHECK:  [[PAIR:%.*]] = cmpxchg ptr %{{.*}}, i32 %{{.*}}, i32 %{{.*}} syncscope("device") monotonic monotonic
// SPVCHECK: [[PAIR:%.*]] = cmpxchg ptr addrspace(11) %{{.*}}, i32 %{{.*}}, i32 %{{.*}} syncscope("device") monotonic monotonic
// CHECK-NEXT: [[OLD:%.*]] = extractvalue { i32, i1 } [[PAIR]], 0
// CHECK-NEXT: store i32 [[OLD]], ptr {{.*}}%orig
export void test_device(uint cmp, uint v) {
  uint orig;
  InterlockedCompareExchange(Buf[0], cmp, v, orig);
}
