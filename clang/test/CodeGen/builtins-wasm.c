// RUN: %clang_cc1 -triple wasm32-unknown-unknown -fno-lax-vector-conversions \
// RUN:   -O3 -emit-llvm -o - %s \
// RUN:   | FileCheck %s -check-prefixes WEBASSEMBLY,WEBASSEMBLY32
// RUN: %clang_cc1 -triple wasm64-unknown-unknown -fno-lax-vector-conversions \
// RUN:   -O3 -emit-llvm -o - %s \
// RUN:   | FileCheck %s -check-prefixes WEBASSEMBLY,WEBASSEMBLY64

// SIMD convenience types
typedef char i8x16 __attribute((vector_size(16)));
typedef short i16x8 __attribute((vector_size(16)));
typedef int i32x4 __attribute((vector_size(16)));
typedef long long i64x2 __attribute((vector_size(16)));
typedef unsigned char u8x16 __attribute((vector_size(16)));
typedef unsigned short u16x8 __attribute((vector_size(16)));
typedef unsigned int u32x4 __attribute((vector_size(16)));
typedef unsigned long long u64x2 __attribute((vector_size(16)));
typedef float f32x4 __attribute((vector_size(16)));
typedef double f64x2 __attribute((vector_size(16)));

__SIZE_TYPE__ f0(void) {
  return __builtin_wasm_memory_size(0);
  // WEBASSEMBLY32: call {{i.*}} @llvm.wasm.memory.size.i32(i32 0)
  // WEBASSEMBLY64: call {{i.*}} @llvm.wasm.memory.size.i64(i32 0)
}

__SIZE_TYPE__ f1(__SIZE_TYPE__ delta) {
  return __builtin_wasm_memory_grow(0, delta);
  // WEBASSEMBLY32: call i32 @llvm.wasm.memory.grow.i32(i32 0, i32 %{{.*}})
  // WEBASSEMBLY64: call i64 @llvm.wasm.memory.grow.i64(i32 0, i64 %{{.*}})
}

__SIZE_TYPE__ f2(void) {
  return __builtin_wasm_mem_size(0);
  // WEBASSEMBLY32: call {{i.*}} @llvm.wasm.mem.size.i32(i32 0)
  // WEBASSEMBLY64: call {{i.*}} @llvm.wasm.mem.size.i64(i32 0)
}

__SIZE_TYPE__ f3(__SIZE_TYPE__ delta) {
  return __builtin_wasm_mem_grow(0, delta);
  // WEBASSEMBLY32: call i32 @llvm.wasm.mem.grow.i32(i32 0, i32 %{{.*}})
  // WEBASSEMBLY64: call i64 @llvm.wasm.mem.grow.i64(i32 0, i64 %{{.*}})
}

__SIZE_TYPE__ f4(void) {
  return __builtin_wasm_current_memory();
  // WEBASSEMBLY32: call {{i.*}} @llvm.wasm.current.memory.i32()
  // WEBASSEMBLY64: call {{i.*}} @llvm.wasm.current.memory.i64()
}

__SIZE_TYPE__ f5(__SIZE_TYPE__ delta) {
  return __builtin_wasm_grow_memory(delta);
  // WEBASSEMBLY32: call i32 @llvm.wasm.grow.memory.i32(i32 %{{.*}})
  // WEBASSEMBLY64: call i64 @llvm.wasm.grow.memory.i64(i64 %{{.*}})
}

void f6(unsigned int tag, void *obj) {
  return __builtin_wasm_throw(tag, obj);
  // WEBASSEMBLY32: call void @llvm.wasm.throw(i32 %{{.*}}, i8* %{{.*}})
  // WEBASSEMBLY64: call void @llvm.wasm.throw(i32 %{{.*}}, i8* %{{.*}})
}

void f7(void) {
  return __builtin_wasm_rethrow();
  // WEBASSEMBLY32: call void @llvm.wasm.rethrow()
  // WEBASSEMBLY64: call void @llvm.wasm.rethrow()
}

int f8(int *addr, int expected, long long timeout) {
  return __builtin_wasm_atomic_wait_i32(addr, expected, timeout);
  // WEBASSEMBLY32: call i32 @llvm.wasm.atomic.wait.i32(i32* %{{.*}}, i32 %{{.*}}, i64 %{{.*}})
  // WEBASSEMBLY64: call i32 @llvm.wasm.atomic.wait.i32(i32* %{{.*}}, i32 %{{.*}}, i64 %{{.*}})
}

int f9(long long *addr, long long expected, long long timeout) {
  return __builtin_wasm_atomic_wait_i64(addr, expected, timeout);
  // WEBASSEMBLY32: call i32 @llvm.wasm.atomic.wait.i64(i64* %{{.*}}, i64 %{{.*}}, i64 %{{.*}})
  // WEBASSEMBLY64: call i32 @llvm.wasm.atomic.wait.i64(i64* %{{.*}}, i64 %{{.*}}, i64 %{{.*}})
}

unsigned int f10(int *addr, int count) {
  return __builtin_wasm_atomic_notify(addr, count);
  // WEBASSEMBLY32: call i32 @llvm.wasm.atomic.notify(i32* %{{.*}}, i32 %{{.*}})
  // WEBASSEMBLY64: call i32 @llvm.wasm.atomic.notify(i32* %{{.*}}, i32 %{{.*}})
}

int f11(i8x16 v) {
  return __builtin_wasm_extract_lane_s_i8x16(v, 13);
  // WEBASSEMBLY: extractelement <16 x i8> %v, i32 13
  // WEBASSEMBLY-NEXT: sext
  // WEBASSEMBLY-NEXT: ret
}

int f12(i8x16 v) {
  return __builtin_wasm_extract_lane_u_i8x16(v, 13);
  // WEBASSEMBLY: extractelement <16 x i8> %v, i32 13
  // WEBASSEMBLY-NEXT: zext
  // WEBASSEMBLY-NEXT: ret
}

int f13(i16x8 v) {
  return __builtin_wasm_extract_lane_s_i16x8(v, 7);
  // WEBASSEMBLY: extractelement <8 x i16> %v, i32 7
  // WEBASSEMBLY-NEXT: sext
  // WEBASSEMBLY-NEXT: ret
}

int f14(i16x8 v) {
  return __builtin_wasm_extract_lane_u_i16x8(v, 7);
  // WEBASSEMBLY: extractelement <8 x i16> %v, i32 7
  // WEBASSEMBLY-NEXT: zext
  // WEBASSEMBLY-NEXT: ret
}

int f15(i32x4 v) {
  return __builtin_wasm_extract_lane_i32x4(v, 3);
  // WEBASSEMBLY: extractelement <4 x i32> %v, i32 3
  // WEBASSEMBLY-NEXT: ret
}

long long f16(i64x2 v) {
  return __builtin_wasm_extract_lane_i64x2(v, 1);
  // WEBASSEMBLY: extractelement <2 x i64> %v, i32 1
  // WEBASSEMBLY-NEXT: ret
}

float f17(f32x4 v) {
  return __builtin_wasm_extract_lane_f32x4(v, 3);
  // WEBASSEMBLY: extractelement <4 x float> %v, i32 3
  // WEBASSEMBLY-NEXT: ret
}

double f18(f64x2 v) {
  return __builtin_wasm_extract_lane_f64x2(v, 1);
  // WEBASSEMBLY: extractelement <2 x double> %v, i32 1
  // WEBASSEMBLY-NEXT: ret
}

i8x16 f19(i8x16 v, int x) {
  return __builtin_wasm_replace_lane_i8x16(v, 13, x);
  // WEBASSEMBLY: trunc i32 %x to i8
  // WEBASSEMBLY-NEXT: insertelement <16 x i8> %v, i8 %{{.*}}, i32 13
  // WEBASSEMBLY-NEXT: ret
}

i16x8 f20(i16x8 v, int x) {
  return __builtin_wasm_replace_lane_i16x8(v, 7, x);
  // WEBASSEMBLY: trunc i32 %x to i16
  // WEBASSEMBLY-NEXT: insertelement <8 x i16> %v, i16 %{{.*}}, i32 7
  // WEBASSEMBLY-NEXT: ret
}

i32x4 f21(i32x4 v, int x) {
  return __builtin_wasm_replace_lane_i32x4(v, 3, x);
  // WEBASSEMBLY: insertelement <4 x i32> %v, i32 %x, i32 3
  // WEBASSEMBLY-NEXT: ret
}

i64x2 f22(i64x2 v, long long x) {
  return __builtin_wasm_replace_lane_i64x2(v, 1, x);
  // WEBASSEMBLY: insertelement <2 x i64> %v, i64 %x, i32 1
  // WEBASSEMBLY-NEXT: ret
}

f32x4 f23(f32x4 v, float x) {
  return __builtin_wasm_replace_lane_f32x4(v, 3, x);
  // WEBASSEMBLY: insertelement <4 x float> %v, float %x, i32 3
  // WEBASSEMBLY-NEXT: ret
}

f64x2 f24(f64x2 v, double x) {
  return __builtin_wasm_replace_lane_f64x2(v, 1, x);
  // WEBASSEMBLY: insertelement <2 x double> %v, double %x, i32 1
  // WEBASSEMBLY-NEXT: ret
}

i8x16 f25(i8x16 x, i8x16 y) {
  return __builtin_wasm_add_saturate_s_i8x16(x, y);
  // WEBASSEMBLY: call <16 x i8> @llvm.wasm.add.saturate.signed.v16i8(
  // WEBASSEMBLY-SAME: <16 x i8> %x, <16 x i8> %y)
  // WEBASSEMBLY-NEXT: ret
}

i8x16 f26(i8x16 x, i8x16 y) {
  return __builtin_wasm_add_saturate_u_i8x16(x, y);
  // WEBASSEMBLY: call <16 x i8> @llvm.wasm.add.saturate.unsigned.v16i8(
  // WEBASSEMBLY-SAME: <16 x i8> %x, <16 x i8> %y)
  // WEBASSEMBLY-NEXT: ret
}

i16x8 f27(i16x8 x, i16x8 y) {
  return __builtin_wasm_add_saturate_s_i16x8(x, y);
  // WEBASSEMBLY: call <8 x i16> @llvm.wasm.add.saturate.signed.v8i16(
  // WEBASSEMBLY-SAME: <8 x i16> %x, <8 x i16> %y)
  // WEBASSEMBLY-NEXT: ret
}

i16x8 f28(i16x8 x, i16x8 y) {
  return __builtin_wasm_add_saturate_u_i16x8(x, y);
  // WEBASSEMBLY: call <8 x i16> @llvm.wasm.add.saturate.unsigned.v8i16(
  // WEBASSEMBLY-SAME: <8 x i16> %x, <8 x i16> %y)
  // WEBASSEMBLY-NEXT: ret
}

i8x16 f29(i8x16 x, i8x16 y) {
  return __builtin_wasm_sub_saturate_s_i8x16(x, y);
  // WEBASSEMBLY: call <16 x i8> @llvm.wasm.sub.saturate.signed.v16i8(
  // WEBASSEMBLY-SAME: <16 x i8> %x, <16 x i8> %y)
  // WEBASSEMBLY-NEXT: ret
}

i8x16 f30(i8x16 x, i8x16 y) {
  return __builtin_wasm_sub_saturate_u_i8x16(x, y);
  // WEBASSEMBLY: call <16 x i8> @llvm.wasm.sub.saturate.unsigned.v16i8(
  // WEBASSEMBLY-SAME: <16 x i8> %x, <16 x i8> %y)
  // WEBASSEMBLY-NEXT: ret
}

i16x8 f31(i16x8 x, i16x8 y) {
  return __builtin_wasm_sub_saturate_s_i16x8(x, y);
  // WEBASSEMBLY: call <8 x i16> @llvm.wasm.sub.saturate.signed.v8i16(
  // WEBASSEMBLY-SAME: <8 x i16> %x, <8 x i16> %y)
  // WEBASSEMBLY-NEXT: ret
}

i16x8 f32(i16x8 x, i16x8 y) {
  return __builtin_wasm_sub_saturate_u_i16x8(x, y);
  // WEBASSEMBLY: call <8 x i16> @llvm.wasm.sub.saturate.unsigned.v8i16(
  // WEBASSEMBLY-SAME: <8 x i16> %x, <8 x i16> %y)
  // WEBASSEMBLY-NEXT: ret
}

int f33(i8x16 x) {
  return __builtin_wasm_any_true_i8x16(x);
  // WEBASSEMBLY: call i32 @llvm.wasm.anytrue.v16i8(<16 x i8> %x)
  // WEBASSEMBLY: ret
}

int f34(i16x8 x) {
  return __builtin_wasm_any_true_i16x8(x);
  // WEBASSEMBLY: call i32 @llvm.wasm.anytrue.v8i16(<8 x i16> %x)
  // WEBASSEMBLY: ret
}

int f35(i32x4 x) {
  return __builtin_wasm_any_true_i32x4(x);
  // WEBASSEMBLY: call i32 @llvm.wasm.anytrue.v4i32(<4 x i32> %x)
  // WEBASSEMBLY: ret
}

int f36(i64x2 x) {
  return __builtin_wasm_any_true_i64x2(x);
  // WEBASSEMBLY: call i32 @llvm.wasm.anytrue.v2i64(<2 x i64> %x)
  // WEBASSEMBLY: ret
}

int f37(i8x16 x) {
  return __builtin_wasm_all_true_i8x16(x);
  // WEBASSEMBLY: call i32 @llvm.wasm.alltrue.v16i8(<16 x i8> %x)
  // WEBASSEMBLY: ret
}

int f38(i16x8 x) {
  return __builtin_wasm_all_true_i16x8(x);
  // WEBASSEMBLY: call i32 @llvm.wasm.alltrue.v8i16(<8 x i16> %x)
  // WEBASSEMBLY: ret
}

int f39(i32x4 x) {
  return __builtin_wasm_all_true_i32x4(x);
  // WEBASSEMBLY: call i32 @llvm.wasm.alltrue.v4i32(<4 x i32> %x)
  // WEBASSEMBLY: ret
}

int f40(i64x2 x) {
  return __builtin_wasm_all_true_i64x2(x);
  // WEBASSEMBLY: call i32 @llvm.wasm.alltrue.v2i64(<2 x i64> %x)
  // WEBASSEMBLY: ret
}

f32x4 f41(f32x4 x) {
  return __builtin_wasm_abs_f32x4(x);
  // WEBASSEMBLY: call <4 x float> @llvm.fabs.v4f32(<4 x float> %x)
  // WEBASSEMBLY: ret
}

f64x2 f42(f64x2 x) {
  return __builtin_wasm_abs_f64x2(x);
  // WEBASSEMBLY: call <2 x double> @llvm.fabs.v2f64(<2 x double> %x)
  // WEBASSEMBLY: ret
}

f32x4 f43(f32x4 x) {
  return __builtin_wasm_sqrt_f32x4(x);
  // WEBASSEMBLY: call <4 x float> @llvm.sqrt.v4f32(<4 x float> %x)
  // WEBASSEMBLY: ret
}

f64x2 f44(f64x2 x) {
  return __builtin_wasm_sqrt_f64x2(x);
  // WEBASSEMBLY: call <2 x double> @llvm.sqrt.v2f64(<2 x double> %x)
  // WEBASSEMBLY: ret
}
