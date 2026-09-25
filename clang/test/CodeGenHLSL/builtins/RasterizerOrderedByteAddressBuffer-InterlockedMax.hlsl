// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-library %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s

// SPIR-V codegen for RasterizerOrderedByteAddressBuffer is not implemented.

RasterizerOrderedByteAddressBuffer ROVB : register(u1);

// CHECK-LABEL: define void @{{.*}}test_rovb_int
// CHECK:     %[[PTR:.*]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i8_1_1t.i32({{.*}})
// CHECK:     %[[R:.*]] = atomicrmw max ptr %[[PTR]], i32 %{{.*}} syncscope("device") monotonic
// CHECK-NOT: store i32 %[[R]]
export void test_rovb_int(uint off, int v) {
  ROVB.InterlockedMax(off, v);
}

// CHECK-LABEL: define void @{{.*}}test_rovb_uint_orig
// CHECK: %[[PTR:.*]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i8_1_1t.i32({{.*}})
// CHECK: %[[R:.*]] = atomicrmw umax ptr %[[PTR]], i32 %{{.*}} syncscope("device") monotonic
// CHECK: store i32 %[[R]], ptr {{.*}}
export void test_rovb_uint_orig(uint off, uint v, out uint orig) {
  ROVB.InterlockedMax(off, v, orig);
}

// CHECK-LABEL: define void @{{.*}}test_rovb_int64
// CHECK:     %[[PTR:.*]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i8_1_1t.i32({{.*}})
// CHECK:     %[[R:.*]] = atomicrmw max ptr %[[PTR]], i64 %{{.*}} syncscope("device") monotonic
// CHECK-NOT: store i64 %[[R]]
export void test_rovb_int64(uint off, int64_t v) {
  ROVB.InterlockedMax64(off, v);
}

// CHECK-LABEL: define void @{{.*}}test_rovb_uint64_orig
// CHECK: %[[PTR:.*]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i8_1_1t.i32({{.*}})
// CHECK: %[[R:.*]] = atomicrmw umax ptr %[[PTR]], i64 %{{.*}} syncscope("device") monotonic
// CHECK: store i64 %[[R]], ptr {{.*}}
export void test_rovb_uint64_orig(uint off, uint64_t v, out uint64_t orig) {
  ROVB.InterlockedMax64(off, v, orig);
}
