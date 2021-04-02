// RUN: %clang_cc1 -triple wasm32-unknown-unknown -target-feature +simd128 -target-feature +nontrapping-fptoint -target-feature +exception-handling -target-feature +bulk-memory -target-feature +atomics -flax-vector-conversions=none -O3 -emit-llvm -o - %s | FileCheck %s -check-prefixes WEBASSEMBLY,WEBASSEMBLY32
// RUN: %clang_cc1 -triple wasm64-unknown-unknown -target-feature +simd128 -target-feature +nontrapping-fptoint -target-feature +exception-handling -target-feature +bulk-memory -target-feature +atomics -flax-vector-conversions=none -O3 -emit-llvm -o - %s | FileCheck %s -check-prefixes WEBASSEMBLY,WEBASSEMBLY64
// RUN: not %clang_cc1 -triple wasm64-unknown-unknown -target-feature +nontrapping-fptoint -target-feature +exception-handling -target-feature +bulk-memory -target-feature +atomics -flax-vector-conversions=none -O3 -emit-llvm -o - %s 2>&1 | FileCheck %s -check-prefixes MISSING-SIMD

// SIMD convenience types
typedef signed char i8x16 __attribute((vector_size(16)));
typedef short i16x8 __attribute((vector_size(16)));
typedef int i32x4 __attribute((vector_size(16)));
typedef long long i64x2 __attribute((vector_size(16)));
typedef unsigned char u8x16 __attribute((vector_size(16)));
typedef unsigned short u16x8 __attribute((vector_size(16)));
typedef unsigned int u32x4 __attribute((vector_size(16)));
typedef unsigned long long u64x2 __attribute((vector_size(16)));
typedef float f32x4 __attribute((vector_size(16)));
typedef double f64x2 __attribute((vector_size(16)));

__SIZE_TYPE__ memory_size(void) {
  return __builtin_wasm_memory_size(0);
  // WEBASSEMBLY32: call {{i.*}} @llvm.wasm.memory.size.i32(i32 0)
  // WEBASSEMBLY64: call {{i.*}} @llvm.wasm.memory.size.i64(i32 0)
}

__SIZE_TYPE__ memory_grow(__SIZE_TYPE__ delta) {
  return __builtin_wasm_memory_grow(0, delta);
  // WEBASSEMBLY32: call i32 @llvm.wasm.memory.grow.i32(i32 0, i32 %{{.*}})
  // WEBASSEMBLY64: call i64 @llvm.wasm.memory.grow.i64(i32 0, i64 %{{.*}})
}

__SIZE_TYPE__ tls_size() {
  return __builtin_wasm_tls_size();
  // WEBASSEMBLY32: call i32 @llvm.wasm.tls.size.i32()
  // WEBASSEMBLY64: call i64 @llvm.wasm.tls.size.i64()
}

__SIZE_TYPE__ tls_align() {
  return __builtin_wasm_tls_align();
  // WEBASSEMBLY32: call i32 @llvm.wasm.tls.align.i32()
  // WEBASSEMBLY64: call i64 @llvm.wasm.tls.align.i64()
}

void *tls_base() {
  return __builtin_wasm_tls_base();
  // WEBASSEMBLY: call i8* @llvm.wasm.tls.base()
}

void throw(void *obj) {
  return __builtin_wasm_throw(0, obj);
  // WEBASSEMBLY32: call void @llvm.wasm.throw(i32 0, i8* %{{.*}})
  // WEBASSEMBLY64: call void @llvm.wasm.throw(i32 0, i8* %{{.*}})
}

void rethrow(void) {
  return __builtin_wasm_rethrow();
  // WEBASSEMBLY32: call void @llvm.wasm.rethrow()
  // WEBASSEMBLY64: call void @llvm.wasm.rethrow()
}

int memory_atomic_wait32(int *addr, int expected, long long timeout) {
  return __builtin_wasm_memory_atomic_wait32(addr, expected, timeout);
  // WEBASSEMBLY32: call i32 @llvm.wasm.memory.atomic.wait32(i32* %{{.*}}, i32 %{{.*}}, i64 %{{.*}})
  // WEBASSEMBLY64: call i32 @llvm.wasm.memory.atomic.wait32(i32* %{{.*}}, i32 %{{.*}}, i64 %{{.*}})
}

int memory_atomic_wait64(long long *addr, long long expected, long long timeout) {
  return __builtin_wasm_memory_atomic_wait64(addr, expected, timeout);
  // WEBASSEMBLY32: call i32 @llvm.wasm.memory.atomic.wait64(i64* %{{.*}}, i64 %{{.*}}, i64 %{{.*}})
  // WEBASSEMBLY64: call i32 @llvm.wasm.memory.atomic.wait64(i64* %{{.*}}, i64 %{{.*}}, i64 %{{.*}})
}

unsigned int memory_atomic_notify(int *addr, unsigned int count) {
  return __builtin_wasm_memory_atomic_notify(addr, count);
  // WEBASSEMBLY32: call i32 @llvm.wasm.memory.atomic.notify(i32* %{{.*}}, i32 %{{.*}})
  // WEBASSEMBLY64: call i32 @llvm.wasm.memory.atomic.notify(i32* %{{.*}}, i32 %{{.*}})
}

int trunc_s_i32_f32(float f) {
  return __builtin_wasm_trunc_s_i32_f32(f);
  // WEBASSEMBLY: call i32 @llvm.wasm.trunc.signed.i32.f32(float %f)
  // WEBASSEMBLY-NEXT: ret
}

int trunc_u_i32_f32(float f) {
  return __builtin_wasm_trunc_u_i32_f32(f);
  // WEBASSEMBLY: call i32 @llvm.wasm.trunc.unsigned.i32.f32(float %f)
  // WEBASSEMBLY-NEXT: ret
}

int trunc_s_i32_f64(double f) {
  return __builtin_wasm_trunc_s_i32_f64(f);
  // WEBASSEMBLY: call i32 @llvm.wasm.trunc.signed.i32.f64(double %f)
  // WEBASSEMBLY-NEXT: ret
}

int trunc_u_i32_f64(double f) {
  return __builtin_wasm_trunc_u_i32_f64(f);
  // WEBASSEMBLY: call i32 @llvm.wasm.trunc.unsigned.i32.f64(double %f)
  // WEBASSEMBLY-NEXT: ret
}

long long trunc_s_i64_f32(float f) {
  return __builtin_wasm_trunc_s_i64_f32(f);
  // WEBASSEMBLY: call i64 @llvm.wasm.trunc.signed.i64.f32(float %f)
  // WEBASSEMBLY-NEXT: ret
}

long long trunc_u_i64_f32(float f) {
  return __builtin_wasm_trunc_u_i64_f32(f);
  // WEBASSEMBLY: call i64 @llvm.wasm.trunc.unsigned.i64.f32(float %f)
  // WEBASSEMBLY-NEXT: ret
}

long long trunc_s_i64_f64(double f) {
  return __builtin_wasm_trunc_s_i64_f64(f);
  // WEBASSEMBLY: call i64 @llvm.wasm.trunc.signed.i64.f64(double %f)
  // WEBASSEMBLY-NEXT: ret
}

long long trunc_u_i64_f64(double f) {
  return __builtin_wasm_trunc_u_i64_f64(f);
  // WEBASSEMBLY: call i64 @llvm.wasm.trunc.unsigned.i64.f64(double %f)
  // WEBASSEMBLY-NEXT: ret
}

int trunc_saturate_s_i32_f32(float f) {
  return __builtin_wasm_trunc_saturate_s_i32_f32(f);
  // WEBASSEMBLY: call i32 @llvm.wasm.trunc.saturate.signed.i32.f32(float %f)
  // WEBASSEMBLY-NEXT: ret
}

int trunc_saturate_u_i32_f32(float f) {
  return __builtin_wasm_trunc_saturate_u_i32_f32(f);
  // WEBASSEMBLY: call i32 @llvm.wasm.trunc.saturate.unsigned.i32.f32(float %f)
  // WEBASSEMBLY-NEXT: ret
}

int trunc_saturate_s_i32_f64(double f) {
  return __builtin_wasm_trunc_saturate_s_i32_f64(f);
  // WEBASSEMBLY: call i32 @llvm.wasm.trunc.saturate.signed.i32.f64(double %f)
  // WEBASSEMBLY-NEXT: ret
}

int trunc_saturate_u_i32_f64(double f) {
  return __builtin_wasm_trunc_saturate_u_i32_f64(f);
  // WEBASSEMBLY: call i32 @llvm.wasm.trunc.saturate.unsigned.i32.f64(double %f)
  // WEBASSEMBLY-NEXT: ret
}

long long trunc_saturate_s_i64_f32(float f) {
  return __builtin_wasm_trunc_saturate_s_i64_f32(f);
  // WEBASSEMBLY: call i64 @llvm.wasm.trunc.saturate.signed.i64.f32(float %f)
  // WEBASSEMBLY-NEXT: ret
}

long long trunc_saturate_u_i64_f32(float f) {
  return __builtin_wasm_trunc_saturate_u_i64_f32(f);
  // WEBASSEMBLY: call i64 @llvm.wasm.trunc.saturate.unsigned.i64.f32(float %f)
  // WEBASSEMBLY-NEXT: ret
}

long long trunc_saturate_s_i64_f64(double f) {
  return __builtin_wasm_trunc_saturate_s_i64_f64(f);
  // WEBASSEMBLY: call i64 @llvm.wasm.trunc.saturate.signed.i64.f64(double %f)
  // WEBASSEMBLY-NEXT: ret
}

long long trunc_saturate_u_i64_f64(double f) {
  return __builtin_wasm_trunc_saturate_u_i64_f64(f);
  // WEBASSEMBLY: call i64 @llvm.wasm.trunc.saturate.unsigned.i64.f64(double %f)
  // WEBASSEMBLY-NEXT: ret
}

float min_f32(float x, float y) {
  return __builtin_wasm_min_f32(x, y);
  // WEBASSEMBLY: call float @llvm.minimum.f32(float %x, float %y)
  // WEBASSEMBLY-NEXT: ret
}

float max_f32(float x, float y) {
  return __builtin_wasm_max_f32(x, y);
  // WEBASSEMBLY: call float @llvm.maximum.f32(float %x, float %y)
  // WEBASSEMBLY-NEXT: ret
}

double min_f64(double x, double y) {
  return __builtin_wasm_min_f64(x, y);
  // WEBASSEMBLY: call double @llvm.minimum.f64(double %x, double %y)
  // WEBASSEMBLY-NEXT: ret
}

double max_f64(double x, double y) {
  return __builtin_wasm_max_f64(x, y);
  // WEBASSEMBLY: call double @llvm.maximum.f64(double %x, double %y)
  // WEBASSEMBLY-NEXT: ret
}

int extract_lane_s_i8x16(i8x16 v) {
  return __builtin_wasm_extract_lane_s_i8x16(v, 13);
  // MISSING-SIMD: error: '__builtin_wasm_extract_lane_s_i8x16' needs target feature simd128
  // WEBASSEMBLY: extractelement <16 x i8> %v, i32 13
  // WEBASSEMBLY-NEXT: sext
  // WEBASSEMBLY-NEXT: ret
}

int extract_lane_u_i8x16(u8x16 v) {
  return __builtin_wasm_extract_lane_u_i8x16(v, 13);
  // WEBASSEMBLY: extractelement <16 x i8> %v, i32 13
  // WEBASSEMBLY-NEXT: zext
  // WEBASSEMBLY-NEXT: ret
}

int extract_lane_s_i16x8(i16x8 v) {
  return __builtin_wasm_extract_lane_s_i16x8(v, 7);
  // WEBASSEMBLY: extractelement <8 x i16> %v, i32 7
  // WEBASSEMBLY-NEXT: sext
  // WEBASSEMBLY-NEXT: ret
}

int extract_lane_u_i16x8(u16x8 v) {
  return __builtin_wasm_extract_lane_u_i16x8(v, 7);
  // WEBASSEMBLY: extractelement <8 x i16> %v, i32 7
  // WEBASSEMBLY-NEXT: zext
  // WEBASSEMBLY-NEXT: ret
}

int extract_lane_i32x4(i32x4 v) {
  return __builtin_wasm_extract_lane_i32x4(v, 3);
  // WEBASSEMBLY: extractelement <4 x i32> %v, i32 3
  // WEBASSEMBLY-NEXT: ret
}

long long extract_lane_i64x2(i64x2 v) {
  return __builtin_wasm_extract_lane_i64x2(v, 1);
  // WEBASSEMBLY: extractelement <2 x i64> %v, i32 1
  // WEBASSEMBLY-NEXT: ret
}

float extract_lane_f32x4(f32x4 v) {
  return __builtin_wasm_extract_lane_f32x4(v, 3);
  // WEBASSEMBLY: extractelement <4 x float> %v, i32 3
  // WEBASSEMBLY-NEXT: ret
}

double extract_lane_f64x2(f64x2 v) {
  return __builtin_wasm_extract_lane_f64x2(v, 1);
  // WEBASSEMBLY: extractelement <2 x double> %v, i32 1
  // WEBASSEMBLY-NEXT: ret
}

i8x16 replace_lane_i8x16(i8x16 v, int x) {
  return __builtin_wasm_replace_lane_i8x16(v, 13, x);
  // WEBASSEMBLY: trunc i32 %x to i8
  // WEBASSEMBLY-NEXT: insertelement <16 x i8> %v, i8 %{{.*}}, i32 13
  // WEBASSEMBLY-NEXT: ret
}

i16x8 replace_lane_i16x8(i16x8 v, int x) {
  return __builtin_wasm_replace_lane_i16x8(v, 7, x);
  // WEBASSEMBLY: trunc i32 %x to i16
  // WEBASSEMBLY-NEXT: insertelement <8 x i16> %v, i16 %{{.*}}, i32 7
  // WEBASSEMBLY-NEXT: ret
}

i32x4 replace_lane_i32x4(i32x4 v, int x) {
  return __builtin_wasm_replace_lane_i32x4(v, 3, x);
  // WEBASSEMBLY: insertelement <4 x i32> %v, i32 %x, i32 3
  // WEBASSEMBLY-NEXT: ret
}

i64x2 replace_lane_i64x2(i64x2 v, long long x) {
  return __builtin_wasm_replace_lane_i64x2(v, 1, x);
  // WEBASSEMBLY: insertelement <2 x i64> %v, i64 %x, i32 1
  // WEBASSEMBLY-NEXT: ret
}

f32x4 replace_lane_f32x4(f32x4 v, float x) {
  return __builtin_wasm_replace_lane_f32x4(v, 3, x);
  // WEBASSEMBLY: insertelement <4 x float> %v, float %x, i32 3
  // WEBASSEMBLY-NEXT: ret
}

f64x2 replace_lane_f64x2(f64x2 v, double x) {
  return __builtin_wasm_replace_lane_f64x2(v, 1, x);
  // WEBASSEMBLY: insertelement <2 x double> %v, double %x, i32 1
  // WEBASSEMBLY-NEXT: ret
}

i8x16 load8_lane(signed char *p, i8x16 v) {
  return __builtin_wasm_load8_lane(p, v, 0);
  // WEBASSEMBLY: tail call <16 x i8> @llvm.wasm.load8.lane(
  // WEBASSEMBLY-SAME: i8* %p, <16 x i8> %v, i32 0)
  // WEBASSEMBLY-NEXT: ret
}

i16x8 load16_lane(short *p, i16x8 v) {
  return __builtin_wasm_load16_lane(p, v, 0);
  // WEBASSEMBLY: tail call <8 x i16> @llvm.wasm.load16.lane(
  // WEBASSEMBLY-SAME: i16* %p, <8 x i16> %v, i32 0)
  // WEBASSEMBLY-NEXT: ret
}

i32x4 load32_lane(int *p, i32x4 v) {
  return __builtin_wasm_load32_lane(p, v, 0);
  // WEBASSEMBLY: tail call <4 x i32> @llvm.wasm.load32.lane(
  // WEBASSEMBLY-SAME: i32* %p, <4 x i32> %v, i32 0)
  // WEBASSEMBLY-NEXT: ret
}

i64x2 load64_lane(long long *p, i64x2 v) {
  return __builtin_wasm_load64_lane(p, v, 0);
  // WEBASSEMBLY: tail call <2 x i64> @llvm.wasm.load64.lane(
  // WEBASSEMBLY-SAME: i64* %p, <2 x i64> %v, i32 0)
  // WEBASSEMBLY-NEXT: ret
}

void store8_lane(signed char *p, i8x16 v) {
  __builtin_wasm_store8_lane(p, v, 0);
  // WEBASSEMBLY: call void @llvm.wasm.store8.lane(
  // WEBASSEMBLY-SAME: i8* %p, <16 x i8> %v, i32 0)
  // WEBASSEMBLY-NEXT: ret
}

void store16_lane(short *p, i16x8 v) {
  __builtin_wasm_store16_lane(p, v, 0);
  // WEBASSEMBLY: call void @llvm.wasm.store16.lane(
  // WEBASSEMBLY-SAME: i16* %p, <8 x i16> %v, i32 0)
  // WEBASSEMBLY-NEXT: ret
}

void store32_lane(int *p, i32x4 v) {
  __builtin_wasm_store32_lane(p, v, 0);
  // WEBASSEMBLY: call void @llvm.wasm.store32.lane(
  // WEBASSEMBLY-SAME: i32* %p, <4 x i32> %v, i32 0)
  // WEBASSEMBLY-NEXT: ret
}

void store64_lane(long long *p, i64x2 v) {
  __builtin_wasm_store64_lane(p, v, 0);
  // WEBASSEMBLY: call void @llvm.wasm.store64.lane(
  // WEBASSEMBLY-SAME: i64* %p, <2 x i64> %v, i32 0)
  // WEBASSEMBLY-NEXT: ret
}

i8x16 add_sat_s_i8x16(i8x16 x, i8x16 y) {
  return __builtin_wasm_add_sat_s_i8x16(x, y);
  // WEBASSEMBLY: call <16 x i8> @llvm.sadd.sat.v16i8(
  // WEBASSEMBLY-SAME: <16 x i8> %x, <16 x i8> %y)
  // WEBASSEMBLY-NEXT: ret
}

u8x16 add_sat_u_i8x16(u8x16 x, u8x16 y) {
  return __builtin_wasm_add_sat_u_i8x16(x, y);
  // WEBASSEMBLY: call <16 x i8> @llvm.uadd.sat.v16i8(
  // WEBASSEMBLY-SAME: <16 x i8> %x, <16 x i8> %y)
  // WEBASSEMBLY-NEXT: ret
}

i16x8 add_sat_s_i16x8(i16x8 x, i16x8 y) {
  return __builtin_wasm_add_sat_s_i16x8(x, y);
  // WEBASSEMBLY: call <8 x i16> @llvm.sadd.sat.v8i16(
  // WEBASSEMBLY-SAME: <8 x i16> %x, <8 x i16> %y)
  // WEBASSEMBLY-NEXT: ret
}

u16x8 add_sat_u_i16x8(u16x8 x, u16x8 y) {
  return __builtin_wasm_add_sat_u_i16x8(x, y);
  // WEBASSEMBLY: call <8 x i16> @llvm.uadd.sat.v8i16(
  // WEBASSEMBLY-SAME: <8 x i16> %x, <8 x i16> %y)
  // WEBASSEMBLY-NEXT: ret
}

i8x16 sub_sat_s_i8x16(i8x16 x, i8x16 y) {
  return __builtin_wasm_sub_sat_s_i8x16(x, y);
  // WEBASSEMBLY: call <16 x i8> @llvm.wasm.sub.sat.signed.v16i8(
  // WEBASSEMBLY-SAME: <16 x i8> %x, <16 x i8> %y)
  // WEBASSEMBLY-NEXT: ret
}

u8x16 sub_sat_u_i8x16(u8x16 x, u8x16 y) {
  return __builtin_wasm_sub_sat_u_i8x16(x, y);
  // WEBASSEMBLY: call <16 x i8> @llvm.wasm.sub.sat.unsigned.v16i8(
  // WEBASSEMBLY-SAME: <16 x i8> %x, <16 x i8> %y)
  // WEBASSEMBLY-NEXT: ret
}

i8x16 abs_i8x16(i8x16 v) {
  return __builtin_wasm_abs_i8x16(v);
  // WEBASSEMBLY: call <16 x i8> @llvm.abs.v16i8(<16 x i8> %v, i1 false)
  // WEBASSEMBLY-NEXT: ret
}

i16x8 abs_i16x8(i16x8 v) {
  return __builtin_wasm_abs_i16x8(v);
  // WEBASSEMBLY: call <8 x i16> @llvm.abs.v8i16(<8 x i16> %v, i1 false)
  // WEBASSEMBLY-NEXT: ret
}

i32x4 abs_i32x4(i32x4 v) {
  return __builtin_wasm_abs_i32x4(v);
  // WEBASSEMBLY: call <4 x i32> @llvm.abs.v4i32(<4 x i32> %v, i1 false)
  // WEBASSEMBLY-NEXT: ret
}

i8x16 min_s_i8x16(i8x16 x, i8x16 y) {
  return __builtin_wasm_min_s_i8x16(x, y);
  // WEBASSEMBLY: %0 = icmp slt <16 x i8> %x, %y
  // WEBASSEMBLY-NEXT: %1 = select <16 x i1> %0, <16 x i8> %x, <16 x i8> %y
  // WEBASSEMBLY-NEXT: ret <16 x i8> %1
}

u8x16 min_u_i8x16(u8x16 x, u8x16 y) {
  return __builtin_wasm_min_u_i8x16(x, y);
  // WEBASSEMBLY: %0 = icmp ult <16 x i8> %x, %y
  // WEBASSEMBLY-NEXT: %1 = select <16 x i1> %0, <16 x i8> %x, <16 x i8> %y
  // WEBASSEMBLY-NEXT: ret <16 x i8> %1
}

i8x16 max_s_i8x16(i8x16 x, i8x16 y) {
  return __builtin_wasm_max_s_i8x16(x, y);
  // WEBASSEMBLY: %0 = icmp sgt <16 x i8> %x, %y
  // WEBASSEMBLY-NEXT: %1 = select <16 x i1> %0, <16 x i8> %x, <16 x i8> %y
  // WEBASSEMBLY-NEXT: ret <16 x i8> %1
}

u8x16 max_u_i8x16(u8x16 x, u8x16 y) {
  return __builtin_wasm_max_u_i8x16(x, y);
  // WEBASSEMBLY: %0 = icmp ugt <16 x i8> %x, %y
  // WEBASSEMBLY-NEXT: %1 = select <16 x i1> %0, <16 x i8> %x, <16 x i8> %y
  // WEBASSEMBLY-NEXT: ret <16 x i8> %1
}

i16x8 min_s_i16x8(i16x8 x, i16x8 y) {
  return __builtin_wasm_min_s_i16x8(x, y);
  // WEBASSEMBLY: %0 = icmp slt <8 x i16> %x, %y
  // WEBASSEMBLY-NEXT: %1 = select <8 x i1> %0, <8 x i16> %x, <8 x i16> %y
  // WEBASSEMBLY-NEXT: ret <8 x i16> %1
}

u16x8 min_u_i16x8(u16x8 x, u16x8 y) {
  return __builtin_wasm_min_u_i16x8(x, y);
  // WEBASSEMBLY: %0 = icmp ult <8 x i16> %x, %y
  // WEBASSEMBLY-NEXT: %1 = select <8 x i1> %0, <8 x i16> %x, <8 x i16> %y
  // WEBASSEMBLY-NEXT: ret <8 x i16> %1
}

i16x8 max_s_i16x8(i16x8 x, i16x8 y) {
  return __builtin_wasm_max_s_i16x8(x, y);
  // WEBASSEMBLY: %0 = icmp sgt <8 x i16> %x, %y
  // WEBASSEMBLY-NEXT: %1 = select <8 x i1> %0, <8 x i16> %x, <8 x i16> %y
  // WEBASSEMBLY-NEXT: ret <8 x i16> %1
}

u16x8 max_u_i16x8(u16x8 x, u16x8 y) {
  return __builtin_wasm_max_u_i16x8(x, y);
  // WEBASSEMBLY: %0 = icmp ugt <8 x i16> %x, %y
  // WEBASSEMBLY-NEXT: %1 = select <8 x i1> %0, <8 x i16> %x, <8 x i16> %y
  // WEBASSEMBLY-NEXT: ret <8 x i16> %1
}

i32x4 min_s_i32x4(i32x4 x, i32x4 y) {
  return __builtin_wasm_min_s_i32x4(x, y);
  // WEBASSEMBLY: %0 = icmp slt <4 x i32> %x, %y
  // WEBASSEMBLY-NEXT: %1 = select <4 x i1> %0, <4 x i32> %x, <4 x i32> %y
  // WEBASSEMBLY-NEXT: ret <4 x i32> %1
}

u32x4 min_u_i32x4(u32x4 x, u32x4 y) {
  return __builtin_wasm_min_u_i32x4(x, y);
  // WEBASSEMBLY: %0 = icmp ult <4 x i32> %x, %y
  // WEBASSEMBLY-NEXT: %1 = select <4 x i1> %0, <4 x i32> %x, <4 x i32> %y
  // WEBASSEMBLY-NEXT: ret <4 x i32> %1
}

i32x4 max_s_i32x4(i32x4 x, i32x4 y) {
  return __builtin_wasm_max_s_i32x4(x, y);
  // WEBASSEMBLY: %0 = icmp sgt <4 x i32> %x, %y
  // WEBASSEMBLY-NEXT: %1 = select <4 x i1> %0, <4 x i32> %x, <4 x i32> %y
  // WEBASSEMBLY-NEXT: ret <4 x i32> %1
}

u32x4 max_u_i32x4(u32x4 x, u32x4 y) {
  return __builtin_wasm_max_u_i32x4(x, y);
  // WEBASSEMBLY: %0 = icmp ugt <4 x i32> %x, %y
  // WEBASSEMBLY-NEXT: %1 = select <4 x i1> %0, <4 x i32> %x, <4 x i32> %y
  // WEBASSEMBLY-NEXT: ret <4 x i32> %1
}

i16x8 sub_sat_s_i16x8(i16x8 x, i16x8 y) {
  return __builtin_wasm_sub_sat_s_i16x8(x, y);
  // WEBASSEMBLY: call <8 x i16> @llvm.wasm.sub.sat.signed.v8i16(
  // WEBASSEMBLY-SAME: <8 x i16> %x, <8 x i16> %y)
  // WEBASSEMBLY-NEXT: ret
}

u16x8 sub_sat_u_i16x8(u16x8 x, u16x8 y) {
  return __builtin_wasm_sub_sat_u_i16x8(x, y);
  // WEBASSEMBLY: call <8 x i16> @llvm.wasm.sub.sat.unsigned.v8i16(
  // WEBASSEMBLY-SAME: <8 x i16> %x, <8 x i16> %y)
  // WEBASSEMBLY-NEXT: ret
}

u8x16 avgr_u_i8x16(u8x16 x, u8x16 y) {
  return __builtin_wasm_avgr_u_i8x16(x, y);
  // WEBASSEMBLY: call <16 x i8> @llvm.wasm.avgr.unsigned.v16i8(
  // WEBASSEMBLY-SAME: <16 x i8> %x, <16 x i8> %y)
  // WEBASSEMBLY-NEXT: ret
}

u16x8 avgr_u_i16x8(u16x8 x, u16x8 y) {
  return __builtin_wasm_avgr_u_i16x8(x, y);
  // WEBASSEMBLY: call <8 x i16> @llvm.wasm.avgr.unsigned.v8i16(
  // WEBASSEMBLY-SAME: <8 x i16> %x, <8 x i16> %y)
  // WEBASSEMBLY-NEXT: ret
}

i16x8 q15mulr_sat_s_i16x8(i16x8 x, i16x8 y) {
  return __builtin_wasm_q15mulr_sat_s_i16x8(x, y);
  // WEBASSEMBLY: call <8 x i16> @llvm.wasm.q15mulr.sat.signed(
  // WEBASSEMBLY-SAME: <8 x i16> %x, <8 x i16> %y)
  // WEBASSEMBLY-NEXT: ret
}

i16x8 extmul_low_i8x16_s_i16x8(i8x16 x, i8x16 y) {
  return __builtin_wasm_extmul_low_i8x16_s_i16x8(x, y);
  // WEBASSEMBLY: call <8 x i16> @llvm.wasm.extmul.low.signed.v8i16(
  // WEBASSEMBLY-SAME: <16 x i8> %x, <16 x i8> %y)
  // WEBASSEMBLY-NEXT: ret
}

i16x8 extmul_high_i8x16_s_i16x8(i8x16 x, i8x16 y) {
  return __builtin_wasm_extmul_high_i8x16_s_i16x8(x, y);
  // WEBASSEMBLY: call <8 x i16> @llvm.wasm.extmul.high.signed.v8i16(
  // WEBASSEMBLY-SAME: <16 x i8> %x, <16 x i8> %y)
  // WEBASSEMBLY-NEXT: ret
}

u16x8 extmul_low_i8x16_u_i16x8(u8x16 x, u8x16 y) {
  return __builtin_wasm_extmul_low_i8x16_u_i16x8(x, y);
  // WEBASSEMBLY: call <8 x i16> @llvm.wasm.extmul.low.unsigned.v8i16(
  // WEBASSEMBLY-SAME: <16 x i8> %x, <16 x i8> %y)
  // WEBASSEMBLY-NEXT: ret
}

u16x8 extmul_high_i8x16_u_i16x8(u8x16 x, u8x16 y) {
  return __builtin_wasm_extmul_high_i8x16_u_i16x8(x, y);
  // WEBASSEMBLY: call <8 x i16> @llvm.wasm.extmul.high.unsigned.v8i16(
  // WEBASSEMBLY-SAME: <16 x i8> %x, <16 x i8> %y)
  // WEBASSEMBLY-NEXT: ret
}

i32x4 extmul_low_i16x8_s_i32x4(i16x8 x, i16x8 y) {
  return __builtin_wasm_extmul_low_i16x8_s_i32x4(x, y);
  // WEBASSEMBLY: call <4 x i32> @llvm.wasm.extmul.low.signed.v4i32(
  // WEBASSEMBLY-SAME: <8 x i16> %x, <8 x i16> %y)
  // WEBASSEMBLY-NEXT: ret
}

i32x4 extmul_high_i16x8_s_i32x4(i16x8 x, i16x8 y) {
  return __builtin_wasm_extmul_high_i16x8_s_i32x4(x, y);
  // WEBASSEMBLY: call <4 x i32> @llvm.wasm.extmul.high.signed.v4i32(
  // WEBASSEMBLY-SAME: <8 x i16> %x, <8 x i16> %y)
  // WEBASSEMBLY-NEXT: ret
}

u32x4 extmul_low_i16x8_u_i32x4(u16x8 x, u16x8 y) {
  return __builtin_wasm_extmul_low_i16x8_u_i32x4(x, y);
  // WEBASSEMBLY: call <4 x i32> @llvm.wasm.extmul.low.unsigned.v4i32(
  // WEBASSEMBLY-SAME: <8 x i16> %x, <8 x i16> %y)
  // WEBASSEMBLY-NEXT: ret
}

u32x4 extmul_high_i16x8_u_i32x4(u16x8 x, u16x8 y) {
  return __builtin_wasm_extmul_high_i16x8_u_i32x4(x, y);
  // WEBASSEMBLY: call <4 x i32> @llvm.wasm.extmul.high.unsigned.v4i32(
  // WEBASSEMBLY-SAME: <8 x i16> %x, <8 x i16> %y)
  // WEBASSEMBLY-NEXT: ret
}

i64x2 extmul_low_i32x4_s_i64x2(i32x4 x, i32x4 y) {
  return __builtin_wasm_extmul_low_i32x4_s_i64x2(x, y);
  // WEBASSEMBLY: call <2 x i64> @llvm.wasm.extmul.low.signed.v2i64(
  // WEBASSEMBLY-SAME: <4 x i32> %x, <4 x i32> %y)
  // WEBASSEMBLY-NEXT: ret
}

i64x2 extmul_high_i32x4_s_i64x2(i32x4 x, i32x4 y) {
  return __builtin_wasm_extmul_high_i32x4_s_i64x2(x, y);
  // WEBASSEMBLY: call <2 x i64> @llvm.wasm.extmul.high.signed.v2i64(
  // WEBASSEMBLY-SAME: <4 x i32> %x, <4 x i32> %y)
  // WEBASSEMBLY-NEXT: ret
}

u64x2 extmul_low_i32x4_u_i64x2(u32x4 x, u32x4 y) {
  return __builtin_wasm_extmul_low_i32x4_u_i64x2(x, y);
  // WEBASSEMBLY: call <2 x i64> @llvm.wasm.extmul.low.unsigned.v2i64(
  // WEBASSEMBLY-SAME: <4 x i32> %x, <4 x i32> %y)
  // WEBASSEMBLY-NEXT: ret
}

u64x2 extmul_high_i32x4_u_i64x2(u32x4 x, u32x4 y) {
  return __builtin_wasm_extmul_high_i32x4_u_i64x2(x, y);
  // WEBASSEMBLY: call <2 x i64> @llvm.wasm.extmul.high.unsigned.v2i64(
  // WEBASSEMBLY-SAME: <4 x i32> %x, <4 x i32> %y)
  // WEBASSEMBLY-NEXT: ret
}

i16x8 extadd_pairwise_i8x16_s_i16x8(i8x16 v) {
  return __builtin_wasm_extadd_pairwise_i8x16_s_i16x8(v);
  // WEBASSEMBLY: call <8 x i16> @llvm.wasm.extadd.pairwise.signed.v8i16(
  // WEBASSEMBLY-SAME: <16 x i8> %v)
  // WEBASSEMBLY-NEXT: ret
}

u16x8 extadd_pairwise_i8x16_u_i16x8(u8x16 v) {
  return __builtin_wasm_extadd_pairwise_i8x16_u_i16x8(v);
  // WEBASSEMBLY: call <8 x i16> @llvm.wasm.extadd.pairwise.unsigned.v8i16(
  // WEBASSEMBLY-SAME: <16 x i8> %v)
  // WEBASSEMBLY-NEXT: ret
}

i32x4 extadd_pairwise_i16x8_s_i32x4(i16x8 v) {
  return __builtin_wasm_extadd_pairwise_i16x8_s_i32x4(v);
  // WEBASSEMBLY: call <4 x i32> @llvm.wasm.extadd.pairwise.signed.v4i32(
  // WEBASSEMBLY-SAME: <8 x i16> %v)
  // WEBASSEMBLY-NEXT: ret
}

u32x4 extadd_pairwise_i16x8_u_i32x4(u16x8 v) {
  return __builtin_wasm_extadd_pairwise_i16x8_u_i32x4(v);
  // WEBASSEMBLY: call <4 x i32> @llvm.wasm.extadd.pairwise.unsigned.v4i32(
  // WEBASSEMBLY-SAME: <8 x i16> %v)
  // WEBASSEMBLY-NEXT: ret
}

i32x4 dot_i16x8_s(i16x8 x, i16x8 y) {
  return __builtin_wasm_dot_s_i32x4_i16x8(x, y);
  // WEBASSEMBLY: call <4 x i32> @llvm.wasm.dot(<8 x i16> %x, <8 x i16> %y)
  // WEBASSEMBLY-NEXT: ret
}

i32x4 bitselect(i32x4 x, i32x4 y, i32x4 c) {
  return __builtin_wasm_bitselect(x, y, c);
  // WEBASSEMBLY: call <4 x i32> @llvm.wasm.bitselect.v4i32(
  // WEBASSEMBLY-SAME: <4 x i32> %x, <4 x i32> %y, <4 x i32> %c)
  // WEBASSEMBLY-NEXT: ret
}

i8x16 popcnt(i8x16 x) {
  return __builtin_wasm_popcnt_i8x16(x);
  // WEBASSEMBLY: call <16 x i8> @llvm.wasm.popcnt(<16 x i8> %x)
  // WEBASSEMBLY-NEXT: ret
}

int any_true_i8x16(i8x16 x) {
  return __builtin_wasm_any_true_i8x16(x);
  // WEBASSEMBLY: call i32 @llvm.wasm.anytrue.v16i8(<16 x i8> %x)
  // WEBASSEMBLY: ret
}

int any_true_i16x8(i16x8 x) {
  return __builtin_wasm_any_true_i16x8(x);
  // WEBASSEMBLY: call i32 @llvm.wasm.anytrue.v8i16(<8 x i16> %x)
  // WEBASSEMBLY: ret
}

int any_true_i32x4(i32x4 x) {
  return __builtin_wasm_any_true_i32x4(x);
  // WEBASSEMBLY: call i32 @llvm.wasm.anytrue.v4i32(<4 x i32> %x)
  // WEBASSEMBLY: ret
}

int any_true_i64x2(i64x2 x) {
  return __builtin_wasm_any_true_i64x2(x);
  // WEBASSEMBLY: call i32 @llvm.wasm.anytrue.v2i64(<2 x i64> %x)
  // WEBASSEMBLY: ret
}

int all_true_i8x16(i8x16 x) {
  return __builtin_wasm_all_true_i8x16(x);
  // WEBASSEMBLY: call i32 @llvm.wasm.alltrue.v16i8(<16 x i8> %x)
  // WEBASSEMBLY: ret
}

int all_true_i16x8(i16x8 x) {
  return __builtin_wasm_all_true_i16x8(x);
  // WEBASSEMBLY: call i32 @llvm.wasm.alltrue.v8i16(<8 x i16> %x)
  // WEBASSEMBLY: ret
}

int all_true_i32x4(i32x4 x) {
  return __builtin_wasm_all_true_i32x4(x);
  // WEBASSEMBLY: call i32 @llvm.wasm.alltrue.v4i32(<4 x i32> %x)
  // WEBASSEMBLY: ret
}

int all_true_i64x2(i64x2 x) {
  return __builtin_wasm_all_true_i64x2(x);
  // WEBASSEMBLY: call i32 @llvm.wasm.alltrue.v2i64(<2 x i64> %x)
  // WEBASSEMBLY: ret
}

int bitmask_i8x16(i8x16 x) {
  return __builtin_wasm_bitmask_i8x16(x);
  // WEBASSEMBLY: call i32 @llvm.wasm.bitmask.v16i8(<16 x i8> %x)
  // WEBASSEMBLY: ret
}

int bitmask_i16x8(i16x8 x) {
  return __builtin_wasm_bitmask_i16x8(x);
  // WEBASSEMBLY: call i32 @llvm.wasm.bitmask.v8i16(<8 x i16> %x)
  // WEBASSEMBLY: ret
}

int bitmask_i32x4(i32x4 x) {
  return __builtin_wasm_bitmask_i32x4(x);
  // WEBASSEMBLY: call i32 @llvm.wasm.bitmask.v4i32(<4 x i32> %x)
  // WEBASSEMBLY: ret
}

int bitmask_i64x2(i64x2 x) {
  return __builtin_wasm_bitmask_i64x2(x);
  // WEBASSEMBLY: call i32 @llvm.wasm.bitmask.v2i64(<2 x i64> %x)
  // WEBASSEMBLY: ret
}

f32x4 abs_f32x4(f32x4 x) {
  return __builtin_wasm_abs_f32x4(x);
  // WEBASSEMBLY: call <4 x float> @llvm.fabs.v4f32(<4 x float> %x)
  // WEBASSEMBLY: ret
}

f64x2 abs_f64x2(f64x2 x) {
  return __builtin_wasm_abs_f64x2(x);
  // WEBASSEMBLY: call <2 x double> @llvm.fabs.v2f64(<2 x double> %x)
  // WEBASSEMBLY: ret
}

f32x4 min_f32x4(f32x4 x, f32x4 y) {
  return __builtin_wasm_min_f32x4(x, y);
  // WEBASSEMBLY: call <4 x float> @llvm.minimum.v4f32(
  // WEBASSEMBLY-SAME: <4 x float> %x, <4 x float> %y)
  // WEBASSEMBLY-NEXT: ret
}

f32x4 max_f32x4(f32x4 x, f32x4 y) {
  return __builtin_wasm_max_f32x4(x, y);
  // WEBASSEMBLY: call <4 x float> @llvm.maximum.v4f32(
  // WEBASSEMBLY-SAME: <4 x float> %x, <4 x float> %y)
  // WEBASSEMBLY-NEXT: ret
}

f32x4 pmin_f32x4(f32x4 x, f32x4 y) {
  return __builtin_wasm_pmin_f32x4(x, y);
  // WEBASSEMBLY: call <4 x float> @llvm.wasm.pmin.v4f32(
  // WEBASSEMBLY-SAME: <4 x float> %x, <4 x float> %y)
  // WEBASSEMBLY-NEXT: ret
}

f32x4 pmax_f32x4(f32x4 x, f32x4 y) {
  return __builtin_wasm_pmax_f32x4(x, y);
  // WEBASSEMBLY: call <4 x float> @llvm.wasm.pmax.v4f32(
  // WEBASSEMBLY-SAME: <4 x float> %x, <4 x float> %y)
  // WEBASSEMBLY-NEXT: ret
}

f64x2 min_f64x2(f64x2 x, f64x2 y) {
  return __builtin_wasm_min_f64x2(x, y);
  // WEBASSEMBLY: call <2 x double> @llvm.minimum.v2f64(
  // WEBASSEMBLY-SAME: <2 x double> %x, <2 x double> %y)
  // WEBASSEMBLY-NEXT: ret
}

f64x2 max_f64x2(f64x2 x, f64x2 y) {
  return __builtin_wasm_max_f64x2(x, y);
  // WEBASSEMBLY: call <2 x double> @llvm.maximum.v2f64(
  // WEBASSEMBLY-SAME: <2 x double> %x, <2 x double> %y)
  // WEBASSEMBLY-NEXT: ret
}

f64x2 pmin_f64x2(f64x2 x, f64x2 y) {
  return __builtin_wasm_pmin_f64x2(x, y);
  // WEBASSEMBLY: call <2 x double> @llvm.wasm.pmin.v2f64(
  // WEBASSEMBLY-SAME: <2 x double> %x, <2 x double> %y)
  // WEBASSEMBLY-NEXT: ret
}

f64x2 pmax_f64x2(f64x2 x, f64x2 y) {
  return __builtin_wasm_pmax_f64x2(x, y);
  // WEBASSEMBLY: call <2 x double> @llvm.wasm.pmax.v2f64(
  // WEBASSEMBLY-SAME: <2 x double> %x, <2 x double> %y)
  // WEBASSEMBLY-NEXT: ret
}

f32x4 ceil_f32x4(f32x4 x) {
  return __builtin_wasm_ceil_f32x4(x);
  // WEBASSEMBLY: call <4 x float> @llvm.wasm.ceil.v4f32(<4 x float> %x)
  // WEBASSEMBLY: ret
}

f32x4 floor_f32x4(f32x4 x) {
  return __builtin_wasm_floor_f32x4(x);
  // WEBASSEMBLY: call <4 x float> @llvm.wasm.floor.v4f32(<4 x float> %x)
  // WEBASSEMBLY: ret
}

f32x4 trunc_f32x4(f32x4 x) {
  return __builtin_wasm_trunc_f32x4(x);
  // WEBASSEMBLY: call <4 x float> @llvm.wasm.trunc.v4f32(<4 x float> %x)
  // WEBASSEMBLY: ret
}

f32x4 nearest_f32x4(f32x4 x) {
  return __builtin_wasm_nearest_f32x4(x);
  // WEBASSEMBLY: call <4 x float> @llvm.wasm.nearest.v4f32(<4 x float> %x)
  // WEBASSEMBLY: ret
}

f64x2 ceil_f64x2(f64x2 x) {
  return __builtin_wasm_ceil_f64x2(x);
  // WEBASSEMBLY: call <2 x double> @llvm.wasm.ceil.v2f64(<2 x double> %x)
  // WEBASSEMBLY: ret
}

f64x2 floor_f64x2(f64x2 x) {
  return __builtin_wasm_floor_f64x2(x);
  // WEBASSEMBLY: call <2 x double> @llvm.wasm.floor.v2f64(<2 x double> %x)
  // WEBASSEMBLY: ret
}

f64x2 trunc_f64x2(f64x2 x) {
  return __builtin_wasm_trunc_f64x2(x);
  // WEBASSEMBLY: call <2 x double> @llvm.wasm.trunc.v2f64(<2 x double> %x)
  // WEBASSEMBLY: ret
}

f64x2 nearest_f64x2(f64x2 x) {
  return __builtin_wasm_nearest_f64x2(x);
  // WEBASSEMBLY: call <2 x double> @llvm.wasm.nearest.v2f64(<2 x double> %x)
  // WEBASSEMBLY: ret
}

f32x4 sqrt_f32x4(f32x4 x) {
  return __builtin_wasm_sqrt_f32x4(x);
  // WEBASSEMBLY: call <4 x float> @llvm.sqrt.v4f32(<4 x float> %x)
  // WEBASSEMBLY: ret
}

f64x2 sqrt_f64x2(f64x2 x) {
  return __builtin_wasm_sqrt_f64x2(x);
  // WEBASSEMBLY: call <2 x double> @llvm.sqrt.v2f64(<2 x double> %x)
  // WEBASSEMBLY: ret
}

i32x4 trunc_saturate_s_i32x4_f32x4(f32x4 f) {
  return __builtin_wasm_trunc_saturate_s_i32x4_f32x4(f);
  // WEBASSEMBLY: call <4 x i32> @llvm.wasm.trunc.saturate.signed.v4i32.v4f32(<4 x float> %f)
  // WEBASSEMBLY-NEXT: ret
}

i32x4 trunc_saturate_u_i32x4_f32x4(f32x4 f) {
  return __builtin_wasm_trunc_saturate_u_i32x4_f32x4(f);
  // WEBASSEMBLY: call <4 x i32> @llvm.wasm.trunc.saturate.unsigned.v4i32.v4f32(<4 x float> %f)
  // WEBASSEMBLY-NEXT: ret
}

i8x16 narrow_s_i8x16_i16x8(i16x8 low, i16x8 high) {
  return __builtin_wasm_narrow_s_i8x16_i16x8(low, high);
  // WEBASSEMBLY: call <16 x i8> @llvm.wasm.narrow.signed.v16i8.v8i16(
  // WEBASSEMBLY-SAME: <8 x i16> %low, <8 x i16> %high)
  // WEBASSEMBLY: ret
}

u8x16 narrow_u_i8x16_i16x8(u16x8 low, u16x8 high) {
  return __builtin_wasm_narrow_u_i8x16_i16x8(low, high);
  // WEBASSEMBLY: call <16 x i8> @llvm.wasm.narrow.unsigned.v16i8.v8i16(
  // WEBASSEMBLY-SAME: <8 x i16> %low, <8 x i16> %high)
  // WEBASSEMBLY: ret
}

i16x8 narrow_s_i16x8_i32x4(i32x4 low, i32x4 high) {
  return __builtin_wasm_narrow_s_i16x8_i32x4(low, high);
  // WEBASSEMBLY: call <8 x i16> @llvm.wasm.narrow.signed.v8i16.v4i32(
  // WEBASSEMBLY-SAME: <4 x i32> %low, <4 x i32> %high)
  // WEBASSEMBLY: ret
}

u16x8 narrow_u_i16x8_i32x4(u32x4 low, u32x4 high) {
  return __builtin_wasm_narrow_u_i16x8_i32x4(low, high);
  // WEBASSEMBLY: call <8 x i16> @llvm.wasm.narrow.unsigned.v8i16.v4i32(
  // WEBASSEMBLY-SAME: <4 x i32> %low, <4 x i32> %high)
  // WEBASSEMBLY: ret
}

i64x2 extend_low_s_i32x4_i64x2(i32x4 x) {
  return __builtin_wasm_extend_low_s_i32x4_i64x2(x);
  // WEBASSEMBLY: call <2 x i64> @llvm.wasm.extend.low.signed(<4 x i32> %x)
  // WEBASSEMBLY: ret
}

i64x2 extend_high_s_i32x4_i64x2(i32x4 x) {
  return __builtin_wasm_extend_high_s_i32x4_i64x2(x);
  // WEBASSEMBLY: call <2 x i64> @llvm.wasm.extend.high.signed(<4 x i32> %x)
  // WEBASSEMBLY: ret
}

u64x2 extend_low_u_i32x4_i64x2(u32x4 x) {
  return __builtin_wasm_extend_low_u_i32x4_i64x2(x);
  // WEBASSEMBLY: call <2 x i64> @llvm.wasm.extend.low.unsigned(<4 x i32> %x)
  // WEBASSEMBLY: ret
}

u64x2 extend_high_u_i32x4_i64x2(u32x4 x) {
  return __builtin_wasm_extend_high_u_i32x4_i64x2(x);
  // WEBASSEMBLY: call <2 x i64> @llvm.wasm.extend.high.unsigned(<4 x i32> %x)
  // WEBASSEMBLY: ret
}

f64x2 convert_low_s_i32x4_f64x2(i32x4 x) {
  return __builtin_wasm_convert_low_s_i32x4_f64x2(x);
  // WEBASSEMBLY: call <2 x double> @llvm.wasm.convert.low.signed(<4 x i32> %x)
  // WEBASSEMBLY: ret
}

f64x2 convert_low_u_i32x4_f64x2(u32x4 x) {
  return __builtin_wasm_convert_low_u_i32x4_f64x2(x);
  // WEBASSEMBLY: call <2 x double> @llvm.wasm.convert.low.unsigned(<4 x i32> %x)
  // WEBASSEMBLY: ret
}

i32x4 trunc_sat_zero_s_f64x2_i32x4(f64x2 x) {
  return __builtin_wasm_trunc_sat_zero_s_f64x2_i32x4(x);
  // WEBASSEMBLY: call <4 x i32> @llvm.wasm.trunc.sat.zero.signed(<2 x double> %x)
  // WEBASSEMBLY: ret
}

u32x4 trunc_sat_zero_u_f64x2_i32x4(f64x2 x) {
  return __builtin_wasm_trunc_sat_zero_u_f64x2_i32x4(x);
  // WEBASSEMBLY: call <4 x i32> @llvm.wasm.trunc.sat.zero.unsigned(<2 x double> %x)
  // WEBASSEMBLY: ret
}

f32x4 wasm_demote_zero_f64x2_f32x4(f64x2 x) {
  return __builtin_wasm_demote_zero_f64x2_f32x4(x);
  // WEBASSEMBLY: call <4 x float> @llvm.wasm.demote.zero(<2 x double> %x)
  // WEBASSEMBLY: ret
}

f64x2 wasm_promote_low_f32x4_f64x2(f32x4 x) {
  return __builtin_wasm_promote_low_f32x4_f64x2(x);
  // WEBASSEMBLY: call <2 x double> @llvm.wasm.promote.low(<4 x float> %x)
  // WEBASSEMBLY: ret
}

i32x4 load32_zero(int *p) {
  return __builtin_wasm_load32_zero(p);
  // WEBASSEMBLY: call <4 x i32> @llvm.wasm.load32.zero(i32* %p)
  // WEBASSEMBLY: ret
}

i64x2 load64_zero(long long *p) {
  return __builtin_wasm_load64_zero(p);
  // WEBASSEMBLY: call <2 x i64> @llvm.wasm.load64.zero(i64* %p)
  // WEBASSEMBLY: ret
}

i8x16 swizzle_v8x16(i8x16 x, i8x16 y) {
  return __builtin_wasm_swizzle_v8x16(x, y);
  // WEBASSEMBLY: call <16 x i8> @llvm.wasm.swizzle(<16 x i8> %x, <16 x i8> %y)
}

i8x16 shuffle(i8x16 x, i8x16 y) {
  return __builtin_wasm_shuffle_v8x16(x, y, 0, 1, 2, 3, 4, 5, 6, 7,
                                      8, 9, 10, 11, 12, 13, 14, 15);
  // WEBASSEMBLY: call <16 x i8> @llvm.wasm.shuffle(<16 x i8> %x, <16 x i8> %y,
  // WEBASSEMBLY-SAME: i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
  // WEBASSEMBLY-SAME: i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14,
  // WEBASSEMBLY-SAME: i32 15
  // WEBASSEMBLY-NEXT: ret
}
