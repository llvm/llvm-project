// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-library %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,DXCHECK

// Test that the RWByteAddressBuffer::InterlockedAdd and
// RasterizerOrderedByteAddressBuffer::InterlockedAdd member methods lower to
// `dx.resource.getpointer.typed -> dx.interlocked.add`, and that the
// 3-argument overload stores the returned original value through the out
// parameter.

RWByteAddressBuffer BAB : register(u0);
RasterizerOrderedByteAddressBuffer ROVB : register(u1);

// CHECK-LABEL: define void @{{.*}}test_bab_int_2arg
// DXCHECK: %[[HANDLE:.*]] = load target("dx.RawBuffer", i8, 1, 0), ptr {{.*}}
// DXCHECK: %[[PTR:.*]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i8_1_0t.i32(target("dx.RawBuffer", i8, 1, 0) %[[HANDLE]], i32 %{{.*}})
// DXCHECK: atomicrmw add ptr %[[PTR]], i32 %{{.*}} monotonic
export void test_bab_int_2arg(uint off, int v) {
  BAB.InterlockedAdd(off, v);
}

// CHECK-LABEL: define void @{{.*}}test_bab_uint_3arg
// DXCHECK: %[[HANDLE:.*]] = load target("dx.RawBuffer", i8, 1, 0), ptr {{.*}}
// DXCHECK: %[[PTR:.*]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i8_1_0t.i32(target("dx.RawBuffer", i8, 1, 0) %[[HANDLE]], i32 %{{.*}})
// DXCHECK: %[[R:.*]] = atomicrmw add ptr %[[PTR]], i32 %{{.*}} monotonic
// DXCHECK: store i32 %[[R]], ptr {{.*}}
export void test_bab_uint_3arg(uint off, uint v, out uint orig) {
  BAB.InterlockedAdd(off, v, orig);
}

// CHECK-LABEL: define void @{{.*}}test_rovb_int_2arg
// DXCHECK: %[[HANDLE:.*]] = load target("dx.RawBuffer", i8, 1, 1), ptr {{.*}}
// DXCHECK: %[[PTR:.*]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i8_1_1t.i32(target("dx.RawBuffer", i8, 1, 1) %[[HANDLE]], i32 %{{.*}})
// DXCHECK: atomicrmw add ptr %[[PTR]], i32 %{{.*}} monotonic
export void test_rovb_int_2arg(uint off, int v) {
  ROVB.InterlockedAdd(off, v);
}

// CHECK-LABEL: define void @{{.*}}test_rovb_uint_3arg
// DXCHECK: %[[HANDLE:.*]] = load target("dx.RawBuffer", i8, 1, 1), ptr {{.*}}
// DXCHECK: %[[PTR:.*]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i8_1_1t.i32(target("dx.RawBuffer", i8, 1, 1) %[[HANDLE]], i32 %{{.*}})
// DXCHECK: %[[R:.*]] = atomicrmw add ptr %[[PTR]], i32 %{{.*}} monotonic
// DXCHECK: store i32 %[[R]], ptr {{.*}}
export void test_rovb_uint_3arg(uint off, uint v, out uint orig) {
  ROVB.InterlockedAdd(off, v, orig);
}

// CHECK-LABEL: define void @{{.*}}test_bab_int64_2arg
// DXCHECK: %[[HANDLE:.*]] = load target("dx.RawBuffer", i8, 1, 0), ptr {{.*}}
// DXCHECK: %[[PTR:.*]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i8_1_0t.i32(target("dx.RawBuffer", i8, 1, 0) %[[HANDLE]], i32 %{{.*}})
// DXCHECK: atomicrmw add ptr %[[PTR]], i64 %{{.*}} monotonic
export void test_bab_int64_2arg(uint off, int64_t v) {
  BAB.InterlockedAdd64(off, v);
}

// CHECK-LABEL: define void @{{.*}}test_bab_uint64_3arg
// DXCHECK: %[[HANDLE:.*]] = load target("dx.RawBuffer", i8, 1, 0), ptr {{.*}}
// DXCHECK: %[[PTR:.*]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i8_1_0t.i32(target("dx.RawBuffer", i8, 1, 0) %[[HANDLE]], i32 %{{.*}})
// DXCHECK: %[[R:.*]] = atomicrmw add ptr %[[PTR]], i64 %{{.*}} monotonic
// DXCHECK: store i64 %[[R]], ptr {{.*}}
export void test_bab_uint64_3arg(uint off, uint64_t v, out uint64_t orig) {
  BAB.InterlockedAdd64(off, v, orig);
}

// CHECK-LABEL: define void @{{.*}}test_rovb_int64_2arg
// DXCHECK: %[[HANDLE:.*]] = load target("dx.RawBuffer", i8, 1, 1), ptr {{.*}}
// DXCHECK: %[[PTR:.*]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i8_1_1t.i32(target("dx.RawBuffer", i8, 1, 1) %[[HANDLE]], i32 %{{.*}})
// DXCHECK: atomicrmw add ptr %[[PTR]], i64 %{{.*}} monotonic
export void test_rovb_int64_2arg(uint off, int64_t v) {
  ROVB.InterlockedAdd64(off, v);
}

// CHECK-LABEL: define void @{{.*}}test_rovb_uint64_3arg
// DXCHECK: %[[HANDLE:.*]] = load target("dx.RawBuffer", i8, 1, 1), ptr {{.*}}
// DXCHECK: %[[PTR:.*]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i8_1_1t.i32(target("dx.RawBuffer", i8, 1, 1) %[[HANDLE]], i32 %{{.*}})
// DXCHECK: %[[R:.*]] = atomicrmw add ptr %[[PTR]], i64 %{{.*}} monotonic
// DXCHECK: store i64 %[[R]], ptr {{.*}}
export void test_rovb_uint64_3arg(uint off, uint64_t v, out uint64_t orig) {
  ROVB.InterlockedAdd64(off, v, orig);
}
