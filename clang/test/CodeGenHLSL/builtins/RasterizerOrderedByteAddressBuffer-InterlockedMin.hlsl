// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-library %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,DXCHECK

// SPIR-V codegen for RasterizerOrderedByteAddressBuffer is not implemented.

RasterizerOrderedByteAddressBuffer ROVB : register(u1);

// CHECK-LABEL: define void @{{.*}}test_rovb_int
// DXCHECK: %[[PTR:.*]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i8_1_1t.i32({{.*}})
// DXCHECK: atomicrmw min ptr %[[PTR]], i32 %{{.*}} syncscope("device") monotonic
export void test_rovb_int(uint off, int v) {
  ROVB.InterlockedMin(off, v);
}

// CHECK-LABEL: define void @{{.*}}test_rovb_uint_orig
// DXCHECK: %[[PTR:.*]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i8_1_1t.i32({{.*}})
// DXCHECK: %[[R:.*]] = atomicrmw umin ptr %[[PTR]], i32 %{{.*}} syncscope("device") monotonic
// DXCHECK: store i32 %[[R]], ptr {{.*}}
export void test_rovb_uint_orig(uint off, uint v, out uint orig) {
  ROVB.InterlockedMin(off, v, orig);
}

// CHECK-LABEL: define void @{{.*}}test_rovb_int64
// DXCHECK: %[[PTR:.*]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i8_1_1t.i32({{.*}})
// DXCHECK: atomicrmw min ptr %[[PTR]], i64 %{{.*}} syncscope("device") monotonic
export void test_rovb_int64(uint off, int64_t v) {
  ROVB.InterlockedMin64(off, v);
}

// CHECK-LABEL: define void @{{.*}}test_rovb_uint64_orig
// DXCHECK: %[[PTR:.*]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i8_1_1t.i32({{.*}})
// DXCHECK: %[[R:.*]] = atomicrmw umin ptr %[[PTR]], i64 %{{.*}} syncscope("device") monotonic
// DXCHECK: store i64 %[[R]], ptr {{.*}}
export void test_rovb_uint64_orig(uint off, uint64_t v, out uint64_t orig) {
  ROVB.InterlockedMin64(off, v, orig);
}
