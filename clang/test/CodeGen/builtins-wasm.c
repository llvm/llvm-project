// RUN: %clang_cc1 -triple wasm32-unknown-unknown -target-feature +simd128 -target-feature +relaxed-simd -target-feature +nontrapping-fptoint -target-feature +exception-handling -target-feature +bulk-memory -target-feature +atomics -flax-vector-conversions=none -O3 -emit-llvm -o - %s | FileCheck %s -check-prefixes WEBASSEMBLY,WEBASSEMBLY32
// RUN: %clang_cc1 -triple wasm64-unknown-unknown -target-feature +simd128 -target-feature +relaxed-simd -target-feature +nontrapping-fptoint -target-feature +exception-handling -target-feature +bulk-memory -target-feature +atomics -flax-vector-conversions=none -O3 -emit-llvm -o - %s | FileCheck %s -check-prefixes WEBASSEMBLY,WEBASSEMBLY64
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
  // WEBASSEMBLY: call i32 @llvm.fptosi.sat.i32.f32(float %f)
  // WEBASSEMBLY-NEXT: ret
}

int trunc_saturate_u_i32_f32(float f) {
  return __builtin_wasm_trunc_saturate_u_i32_f32(f);
  // WEBASSEMBLY: call i32 @llvm.fptoui.sat.i32.f32(float %f)
  // WEBASSEMBLY-NEXT: ret
}

int trunc_saturate_s_i32_f64(double f) {
  return __builtin_wasm_trunc_saturate_s_i32_f64(f);
  // WEBASSEMBLY: call i32 @llvm.fptosi.sat.i32.f64(double %f)
  // WEBASSEMBLY-NEXT: ret
}

int trunc_saturate_u_i32_f64(double f) {
  return __builtin_wasm_trunc_saturate_u_i32_f64(f);
  // WEBASSEMBLY: call i32 @llvm.fptoui.sat.i32.f64(double %f)
  // WEBASSEMBLY-NEXT: ret
}

long long trunc_saturate_s_i64_f32(float f) {
  return __builtin_wasm_trunc_saturate_s_i64_f32(f);
  // WEBASSEMBLY: call i64 @llvm.fptosi.sat.i64.f32(float %f)
  // WEBASSEMBLY-NEXT: ret
}

long long trunc_saturate_u_i64_f32(float f) {
  return __builtin_wasm_trunc_saturate_u_i64_f32(f);
  // WEBASSEMBLY: call i64 @llvm.fptoui.sat.i64.f32(float %f)
  // WEBASSEMBLY-NEXT: ret
}

long long trunc_saturate_s_i64_f64(double f) {
  return __builtin_wasm_trunc_saturate_s_i64_f64(f);
  // WEBASSEMBLY: call i64 @llvm.fptosi.sat.i64.f64(double %f)
  // WEBASSEMBLY-NEXT: ret
}

long long trunc_saturate_u_i64_f64(double f) {
  return __builtin_wasm_trunc_saturate_u_i64_f64(f);
  // WEBASSEMBLY: call i64 @llvm.fptoui.sat.i64.f64(double %f)
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

i8x16 add_sat_s_i8x16(i8x16 x, i8x16 y) {
  return __builtin_wasm_add_sat_s_i8x16(x, y);
  // MISSING-SIMD: error: '__builtin_wasm_add_sat_s_i8x16' needs target feature simd128
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

i64x2 abs_i64x2(i64x2 v) {
  return __builtin_wasm_abs_i64x2(v);
  // WEBASSEMBLY: call <2 x i64> @llvm.abs.v2i64(<2 x i64> %v, i1 false)
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
  // WEBASSEMBLY: call <16 x i8> @llvm.ctpop.v16i8(<16 x i8> %x)
  // WEBASSEMBLY-NEXT: ret
}

int any_true_v128(i8x16 x) {
  return __builtin_wasm_any_true_v128(x);
  // WEBASSEMBLY: call i32 @llvm.wasm.anytrue.v16i8(<16 x i8> %x)
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
  // WEBASSEMBLY: call <4 x float> @llvm.ceil.v4f32(<4 x float> %x)
  // WEBASSEMBLY: ret
}

f32x4 floor_f32x4(f32x4 x) {
  return __builtin_wasm_floor_f32x4(x);
  // WEBASSEMBLY: call <4 x float> @llvm.floor.v4f32(<4 x float> %x)
  // WEBASSEMBLY: ret
}

f32x4 trunc_f32x4(f32x4 x) {
  return __builtin_wasm_trunc_f32x4(x);
  // WEBASSEMBLY: call <4 x float> @llvm.trunc.v4f32(<4 x float> %x)
  // WEBASSEMBLY: ret
}

f32x4 nearest_f32x4(f32x4 x) {
  return __builtin_wasm_nearest_f32x4(x);
  // WEBASSEMBLY: call <4 x float> @llvm.nearbyint.v4f32(<4 x float> %x)
  // WEBASSEMBLY: ret
}

f64x2 ceil_f64x2(f64x2 x) {
  return __builtin_wasm_ceil_f64x2(x);
  // WEBASSEMBLY: call <2 x double> @llvm.ceil.v2f64(<2 x double> %x)
  // WEBASSEMBLY: ret
}

f64x2 floor_f64x2(f64x2 x) {
  return __builtin_wasm_floor_f64x2(x);
  // WEBASSEMBLY: call <2 x double> @llvm.floor.v2f64(<2 x double> %x)
  // WEBASSEMBLY: ret
}

f64x2 trunc_f64x2(f64x2 x) {
  return __builtin_wasm_trunc_f64x2(x);
  // WEBASSEMBLY: call <2 x double> @llvm.trunc.v2f64(<2 x double> %x)
  // WEBASSEMBLY: ret
}

f64x2 nearest_f64x2(f64x2 x) {
  return __builtin_wasm_nearest_f64x2(x);
  // WEBASSEMBLY: call <2 x double> @llvm.nearbyint.v2f64(<2 x double> %x)
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
  // WEBASSEMBLY: call <4 x i32> @llvm.fptosi.sat.v4i32.v4f32(<4 x float> %f)
  // WEBASSEMBLY-NEXT: ret
}

i32x4 trunc_saturate_u_i32x4_f32x4(f32x4 f) {
  return __builtin_wasm_trunc_saturate_u_i32x4_f32x4(f);
  // WEBASSEMBLY: call <4 x i32> @llvm.fptoui.sat.v4i32.v4f32(<4 x float> %f)
  // WEBASSEMBLY-NEXT: ret
}

i8x16 narrow_s_i8x16_i16x8(i16x8 low, i16x8 high) {
  return __builtin_wasm_narrow_s_i8x16_i16x8(low, high);
  // WEBASSEMBLY: call <16 x i8> @llvm.wasm.narrow.signed.v16i8.v8i16(
  // WEBASSEMBLY-SAME: <8 x i16> %low, <8 x i16> %high)
  // WEBASSEMBLY: ret
}

u8x16 narrow_u_i8x16_i16x8(i16x8 low, i16x8 high) {
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

u16x8 narrow_u_i16x8_i32x4(i32x4 low, i32x4 high) {
  return __builtin_wasm_narrow_u_i16x8_i32x4(low, high);
  // WEBASSEMBLY: call <8 x i16> @llvm.wasm.narrow.unsigned.v8i16.v4i32(
  // WEBASSEMBLY-SAME: <4 x i32> %low, <4 x i32> %high)
  // WEBASSEMBLY: ret
}

i32x4 trunc_sat_zero_s_f64x2_i32x4(f64x2 x) {
  return __builtin_wasm_trunc_sat_zero_s_f64x2_i32x4(x);
  // WEBASSEMBLY: %0 = tail call <2 x i32> @llvm.fptosi.sat.v2i32.v2f64(<2 x double> %x)
  // WEBASSEMBLY: %1 = shufflevector <2 x i32> %0, <2 x i32> zeroinitializer, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // WEBASSEMBLY: ret <4 x i32> %1
}

u32x4 trunc_sat_zero_u_f64x2_i32x4(f64x2 x) {
  return __builtin_wasm_trunc_sat_zero_u_f64x2_i32x4(x);
  // WEBASSEMBLY: %0 = tail call <2 x i32> @llvm.fptoui.sat.v2i32.v2f64(<2 x double> %x)
  // WEBASSEMBLY: %1 = shufflevector <2 x i32> %0, <2 x i32> zeroinitializer, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // WEBASSEMBLY: ret <4 x i32> %1
}

i8x16 swizzle_i8x16(i8x16 x, i8x16 y) {
  return __builtin_wasm_swizzle_i8x16(x, y);
  // WEBASSEMBLY: call <16 x i8> @llvm.wasm.swizzle(<16 x i8> %x, <16 x i8> %y)
}

i8x16 shuffle(i8x16 x, i8x16 y) {
  return __builtin_wasm_shuffle_i8x16(x, y, 0, 1, 2, 3, 4, 5, 6, 7,
                                      8, 9, 10, 11, 12, 13, 14, 15);
  // WEBASSEMBLY: call <16 x i8> @llvm.wasm.shuffle(<16 x i8> %x, <16 x i8> %y,
  // WEBASSEMBLY-SAME: i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
  // WEBASSEMBLY-SAME: i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14,
  // WEBASSEMBLY-SAME: i32 15
  // WEBASSEMBLY-NEXT: ret
}

f32x4 fma_f32x4(f32x4 a, f32x4 b, f32x4 c) {
  return __builtin_wasm_fma_f32x4(a, b, c);
  // WEBASSEMBLY: call <4 x float> @llvm.wasm.fma.v4f32(
  // WEBASSEMBLY-SAME: <4 x float> %a, <4 x float> %b, <4 x float> %c)
  // WEBASSEMBLY-NEXT: ret
}

f32x4 fms_f32x4(f32x4 a, f32x4 b, f32x4 c) {
  return __builtin_wasm_fms_f32x4(a, b, c);
  // WEBASSEMBLY: call <4 x float> @llvm.wasm.fms.v4f32(
  // WEBASSEMBLY-SAME: <4 x float> %a, <4 x float> %b, <4 x float> %c)
  // WEBASSEMBLY-NEXT: ret
}

f64x2 fma_f64x2(f64x2 a, f64x2 b, f64x2 c) {
  return __builtin_wasm_fma_f64x2(a, b, c);
  // WEBASSEMBLY: call <2 x double> @llvm.wasm.fma.v2f64(
  // WEBASSEMBLY-SAME: <2 x double> %a, <2 x double> %b, <2 x double> %c)
  // WEBASSEMBLY-NEXT: ret
}

f64x2 fms_f64x2(f64x2 a, f64x2 b, f64x2 c) {
  return __builtin_wasm_fms_f64x2(a, b, c);
  // WEBASSEMBLY: call <2 x double> @llvm.wasm.fms.v2f64(
  // WEBASSEMBLY-SAME: <2 x double> %a, <2 x double> %b, <2 x double> %c)
  // WEBASSEMBLY-NEXT: ret
}

i8x16 laneselect_i8x16(i8x16 a, i8x16 b, i8x16 c) {
  return __builtin_wasm_laneselect_i8x16(a, b, c);
  // WEBASSEMBLY: call <16 x i8> @llvm.wasm.laneselect.v16i8(
  // WEBASSEMBLY-SAME: <16 x i8> %a, <16 x i8> %b, <16 x i8> %c)
  // WEBASSEMBLY-NEXT: ret
}

i16x8 laneselect_i16x8(i16x8 a, i16x8 b, i16x8 c) {
  return __builtin_wasm_laneselect_i16x8(a, b, c);
  // WEBASSEMBLY: call <8 x i16> @llvm.wasm.laneselect.v8i16(
  // WEBASSEMBLY-SAME: <8 x i16> %a, <8 x i16> %b, <8 x i16> %c)
  // WEBASSEMBLY-NEXT: ret
}

i32x4 laneselect_i32x4(i32x4 a, i32x4 b, i32x4 c) {
  return __builtin_wasm_laneselect_i32x4(a, b, c);
  // WEBASSEMBLY: call <4 x i32> @llvm.wasm.laneselect.v4i32(
  // WEBASSEMBLY-SAME: <4 x i32> %a, <4 x i32> %b, <4 x i32> %c)
  // WEBASSEMBLY-NEXT: ret
}

i64x2 laneselect_i64x2(i64x2 a, i64x2 b, i64x2 c) {
  return __builtin_wasm_laneselect_i64x2(a, b, c);
  // WEBASSEMBLY: call <2 x i64> @llvm.wasm.laneselect.v2i64(
  // WEBASSEMBLY-SAME: <2 x i64> %a, <2 x i64> %b, <2 x i64> %c)
  // WEBASSEMBLY-NEXT: ret
}

i8x16 relaxed_swizzle_i8x16(i8x16 x, i8x16 y) {
  return __builtin_wasm_relaxed_swizzle_i8x16(x, y);
  // WEBASSEMBLY: call <16 x i8> @llvm.wasm.relaxed.swizzle(<16 x i8> %x, <16 x i8> %y)
}
