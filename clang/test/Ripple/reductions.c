// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clang -Wpedantic -S -O1 -emit-llvm %s -o - 2>&1 | FileCheck %s --check-prefixes=CHECKALL,CHECKCLANGGEN --implicit-check-not="warning:" --implicit-check-not="error:"
// RUN: %clang -x c++ -Wpedantic -S -O1 -emit-llvm %s -o - 2>&1 | FileCheck %s --check-prefixes=CHECKALL,CHECKCLANGGEN --implicit-check-not="warning:" --implicit-check-not="error:"
// RUN: %clang -fenable-ripple -Wpedantic -S -O1 -emit-llvm %s -o - 2>&1 | FileCheck %s --check-prefixes=CHECKALL,CHECKRIPPLEGEN --implicit-check-not="warning:" --implicit-check-not="error:"
// RUN: %clang -fenable-ripple -x c++ -Wpedantic -S -O1 -emit-llvm %s -o - 2>&1 | FileCheck %s --check-prefixes=CHECKALL,CHECKRIPPLEGEN --implicit-check-not="warning:" --implicit-check-not="error:"

#include <ripple.h>

#ifdef __cpluplus
extern "C" {
#endif

void check_reduceadd_u8(uint8_t a[128], uint8_t *OutPtr) {
  // CHECKALL: check_reduceadd_u8
  ripple_block_t BS = ripple_set_block_shape(0, 128);
  int idx_x = ripple_id(BS, 0);
  uint8_t tmp = a[idx_x];
  uint8_t out = ripple_reduceadd(0x1, tmp);
  // CHECKCLANGGEN: @llvm.ripple.reduce.add.i8(i64 1, i8
  // CHECKRIPPLEGEN: call i8 @llvm.vector.reduce.add.v128i8
  OutPtr[0] = out;
}

#define gen_reduce_test(N, OP, LONGTYPE, SHORTTYPE)                            \
  void check_reduction_##OP##_##SHORTTYPE(LONGTYPE arg[N], LONGTYPE *OutPtr) { \
    ripple_block_t BS = ripple_set_block_shape(0, N);                          \
    int idx_x = ripple_id(BS, 0);                                              \
    LONGTYPE tmp = arg[idx_x];                                                 \
    LONGTYPE tmp2 = arg[idx_x + N];                                            \
    LONGTYPE out = ripple_reduce##OP(0x1, tmp2);                               \
    LONGTYPE out_rval = ripple_reduce##OP(0x1, (LONGTYPE)(tmp + (LONGTYPE)3)); \
    OutPtr[0] = out / out_rval;                                                \
  }

typedef signed char sc;
typedef unsigned char uc;
typedef signed short ss;
typedef unsigned short us;
typedef signed int si;
typedef unsigned int ui;
typedef signed long sl;
typedef unsigned long ul;
typedef signed long long sll;
typedef unsigned long long ull;

// {{{ ADD

gen_reduce_test (128, add, uint8_t, u8)
// CHECKALL: check_reduction_add_u8
// CHECKCLANGGEN-COUNT-2: @llvm.ripple.reduce.add.i8(i64 1, i8
// CHECKRIPPLEGEN-COUNT-2: call i8 @llvm.vector.reduce.add.v128i8

gen_reduce_test (128, add, int8_t, i8)
// CHECKALL: check_reduction_add_i8
// CHECKCLANGGEN-COUNT-2: @llvm.ripple.reduce.add.i8(i64 1, i8
// CHECKRIPPLEGEN-COUNT-2: call i8 @llvm.vector.reduce.add.v128i8

gen_reduce_test (64, add, uint16_t, u16)
// CHECKALL: check_reduction_add_u16
// CHECKCLANGGEN-COUNT-2: @llvm.ripple.reduce.add.i16(i64 1, i16
// CHECKRIPPLEGEN-COUNT-2: call i16 @llvm.vector.reduce.add.v64i16

gen_reduce_test (64, add, int16_t, i16)
// CHECKALL: check_reduction_add_i16
// CHECKCLANGGEN-COUNT-2: @llvm.ripple.reduce.add.i16(i64 1, i16
// CHECKRIPPLEGEN-COUNT-2: call i16 @llvm.vector.reduce.add.v64i16

gen_reduce_test (32, add, uint32_t, u32)
// CHECKALL: check_reduction_add_u32
// CHECKCLANGGEN-COUNT-2: @llvm.ripple.reduce.add.i32(i64 1, i32
// CHECKRIPPLEGEN-COUNT-2: call i32 @llvm.vector.reduce.add.v32i32

gen_reduce_test (32, add, int32_t, i32)
// CHECKALL: check_reduction_add_i32
// CHECKCLANGGEN-COUNT-2: @llvm.ripple.reduce.add.i32(i64 1, i32
// CHECKRIPPLEGEN-COUNT-2: call i32 @llvm.vector.reduce.add.v32i32

gen_reduce_test (16, add, uint64_t, u64)
// CHECKALL: check_reduction_add_u64
// CHECKCLANGGEN-COUNT-2: @llvm.ripple.reduce.add.i64(i64 1, i64
// CHECKRIPPLEGEN-COUNT-2: call i64 @llvm.vector.reduce.add.v16i64

gen_reduce_test (16, add, int64_t, i64)
// CHECKALL: check_reduction_add_i64
// CHECKCLANGGEN-COUNT-2: @llvm.ripple.reduce.add.i64(i64 1, i64
// CHECKRIPPLEGEN-COUNT-2: call i64 @llvm.vector.reduce.add.v16i64

gen_reduce_test (32, add, float, f32)
// CHECKALL: check_reduction_add_f32
// CHECKCLANGGEN-COUNT-2: call{{.*}}reassoc{{.*}}float @llvm.ripple.reduce.fadd.f32(i64 1, float
// CHECKRIPPLEGEN-COUNT-2: call reassoc float @llvm.vector.reduce.fadd.v32f32

gen_reduce_test (16, add, double, f64)
// CHECKALL: check_reduction_add_f64
// CHECKCLANGGEN-COUNT-2: call{{.*}}reassoc{{.*}}double @llvm.ripple.reduce.fadd.f64(i64 1, double
// CHECKRIPPLEGEN-COUNT-2: call reassoc double @llvm.vector.reduce.fadd.v16f64

gen_reduce_test (128, add, char, c)
gen_reduce_test (128, add, sc, sc)
gen_reduce_test (128, add, uc, uc)
gen_reduce_test (128, add, ss, ss)
gen_reduce_test (128, add, us, us)
gen_reduce_test (128, add, si, si)
gen_reduce_test (128, add, ui, ui)
gen_reduce_test (128, add, sl, sl)
gen_reduce_test (128, add, ul, ul)
gen_reduce_test (128, add, sll, sll)
gen_reduce_test (128, add, ull, ull)

// }}}

// {{{ MAX

gen_reduce_test (128, max, uint8_t, u8)
// CHECKALL: check_reduction_max_u8
// CHECKCLANGGEN-COUNT-2: @llvm.ripple.reduce.umax.i8(i64 1, i8
// CHECKRIPPLEGEN-COUNT-2: call i8 @llvm.vector.reduce.umax.v128i8(

gen_reduce_test (128, max, int8_t, i8)
// CHECKALL: check_reduction_max_i8
// CHECKCLANGGEN-COUNT-2: @llvm.ripple.reduce.smax.i8(i64 1, i8
// CHECKRIPPLEGEN-COUNT-2: call i8 @llvm.vector.reduce.smax.v128i8(

gen_reduce_test (64, max, uint16_t, u16)
// CHECKALL: check_reduction_max_u16
// CHECKCLANGGEN-COUNT-2: @llvm.ripple.reduce.umax.i16(i64 1, i16
// CHECKRIPPLEGEN-COUNT-2: call i16 @llvm.vector.reduce.umax.v64i16(

gen_reduce_test (64, max, int16_t, i16)
// CHECKALL: check_reduction_max_i16
// CHECKCLANGGEN-COUNT-2: @llvm.ripple.reduce.smax.i16(i64 1, i16
// CHECKRIPPLEGEN-COUNT-2: call i16 @llvm.vector.reduce.smax.v64i16(

gen_reduce_test (32, max, uint32_t, u32)
// CHECKALL: check_reduction_max_u32
// CHECKCLANGGEN-COUNT-2: @llvm.ripple.reduce.umax.i32(i64 1, i32
// CHECKRIPPLEGEN-COUNT-2: call i32 @llvm.vector.reduce.umax.v32i32(

gen_reduce_test (32, max, int32_t, i32)
// CHECKALL: check_reduction_max_i32
// CHECKCLANGGEN-COUNT-2: @llvm.ripple.reduce.smax.i32(i64 1, i32
// CHECKRIPPLEGEN-COUNT-2: call i32 @llvm.vector.reduce.smax.v32i32(

gen_reduce_test (16, max, uint64_t, u64)
// CHECKALL: check_reduction_max_u64
// CHECKCLANGGEN-COUNT-2: @llvm.ripple.reduce.umax.i64(i64 1, i64
// CHECKRIPPLEGEN-COUNT-2: call i64 @llvm.vector.reduce.umax.v16i64(

gen_reduce_test (16, max, int64_t, i64)
// CHECKALL: check_reduction_max_i64
// CHECKCLANGGEN-COUNT-2: @llvm.ripple.reduce.smax.i64(i64 1, i64
// CHECKRIPPLEGEN-COUNT-2: call i64 @llvm.vector.reduce.smax.v16i64(

gen_reduce_test (32, max, float, f32)
// CHECKALL: check_reduction_max_f32
// CHECKCLANGGEN-COUNT-2: call{{.*}}reassoc{{.*}}float @llvm.ripple.reduce.fmax.f32(i64 1, float
// CHECKRIPPLEGEN-COUNT-2: call reassoc float @llvm.vector.reduce.fmax.v32f32(

gen_reduce_test (16, max, double, f64)
// CHECKALL: check_reduction_max_f64
// CHECKCLANGGEN-COUNT-2: call{{.*}}reassoc{{.*}}double @llvm.ripple.reduce.fmax.f64(i64 1, double
// CHECKRIPPLEGEN-COUNT-2: call reassoc double @llvm.vector.reduce.fmax.v16f64(

gen_reduce_test (32, maximum, float, f32)
// CHECKALL: check_reduction_maximum_f32
// CHECKCLANGGEN-COUNT-2: call{{.*}}reassoc{{.*}}float @llvm.ripple.reduce.fmaximum.f32(i64 1, float
// CHECKRIPPLEGEN-COUNT-2: call reassoc float @llvm.vector.reduce.fmaximum.v32f32(

gen_reduce_test (16, maximum, double, f64)
// CHECKALL: check_reduction_maximum_f64
// CHECKCLANGGEN-COUNT-2: call{{.*}}reassoc{{.*}}double @llvm.ripple.reduce.fmaximum.f64(i64 1, double
// CHECKRIPPLEGEN-COUNT-2: call reassoc double @llvm.vector.reduce.fmaximum.v16f64(

gen_reduce_test (128, max, char, c)
gen_reduce_test (128, max, sc, sc)
gen_reduce_test (128, max, uc, uc)
gen_reduce_test (128, max, ss, ss)
gen_reduce_test (128, max, us, us)
gen_reduce_test (128, max, si, si)
gen_reduce_test (128, max, ui, ui)
gen_reduce_test (128, max, sl, sl)
gen_reduce_test (128, max, ul, ul)
gen_reduce_test (128, max, sll, sll)
gen_reduce_test (128, max, ull, ull)

// }}}

// {{{ MIN

gen_reduce_test (128, min, uint8_t, u8)
// CHECKALL: check_reduction_min_u8
// CHECKCLANGGEN-COUNT-2: @llvm.ripple.reduce.umin.i8(i64 1, i8
// CHECKRIPPLEGEN-COUNT-2: call i8 @llvm.vector.reduce.umin.v128i8(

gen_reduce_test (128, min, int8_t, i8)
// CHECKALL: check_reduction_min_i8
// CHECKCLANGGEN-COUNT-2: @llvm.ripple.reduce.smin.i8(i64 1, i8
// CHECKRIPPLEGEN-COUNT-2: call i8 @llvm.vector.reduce.smin.v128i8(

gen_reduce_test (64, min, uint16_t, u16)
// CHECKALL: check_reduction_min_u16
// CHECKCLANGGEN-COUNT-2: @llvm.ripple.reduce.umin.i16(i64 1, i16
// CHECKRIPPLEGEN-COUNT-2: call i16 @llvm.vector.reduce.umin.v64i16(

gen_reduce_test (64, min, int16_t, i16)
// CHECKALL: check_reduction_min_i16
// CHECKCLANGGEN-COUNT-2: @llvm.ripple.reduce.smin.i16(i64 1, i16
// CHECKRIPPLEGEN-COUNT-2: call i16 @llvm.vector.reduce.smin.v64i16(

gen_reduce_test (32, min, uint32_t, u32)
// CHECKALL: check_reduction_min_u32
// CHECKCLANGGEN-COUNT-2: @llvm.ripple.reduce.umin.i32(i64 1, i32
// CHECKRIPPLEGEN-COUNT-2: call i32 @llvm.vector.reduce.umin.v32i32(

gen_reduce_test (32, min, int32_t, i32)
// CHECKALL: check_reduction_min_i32
// CHECKCLANGGEN-COUNT-2: @llvm.ripple.reduce.smin.i32(i64 1, i32
// CHECKRIPPLEGEN-COUNT-2: call i32 @llvm.vector.reduce.smin.v32i32(

gen_reduce_test (16, min, uint64_t, u64)
// CHECKALL: check_reduction_min_u64
// CHECKCLANGGEN-COUNT-2: @llvm.ripple.reduce.umin.i64(i64 1, i64
// CHECKRIPPLEGEN-COUNT-2: call i64 @llvm.vector.reduce.umin.v16i64(

gen_reduce_test (16, min, int64_t, i64)
// CHECKALL: check_reduction_min_i64
// CHECKCLANGGEN-COUNT-2: @llvm.ripple.reduce.smin.i64(i64 1, i64
// CHECKRIPPLEGEN-COUNT-2: call i64 @llvm.vector.reduce.smin.v16i64(

gen_reduce_test (32, min, float, f32)
// CHECKALL: check_reduction_min_f32
// CHECKCLANGGEN-COUNT-2: call{{.*}}reassoc{{.*}}float @llvm.ripple.reduce.fmin.f32(i64 1, float
// CHECKRIPPLEGEN-COUNT-2: call reassoc float @llvm.vector.reduce.fmin.v32f32(

gen_reduce_test (16, min, double, f64)
// CHECKALL: check_reduction_min_f64
// CHECKCLANGGEN-COUNT-2: call{{.*}}reassoc{{.*}}double @llvm.ripple.reduce.fmin.f64(i64 1, double
// CHECKRIPPLEGEN-COUNT-2: call reassoc double @llvm.vector.reduce.fmin.v16f64(

gen_reduce_test (32, minimum, float, f32)
// CHECKALL: check_reduction_minimum_f32
// CHECKCLANGGEN-COUNT-2: call{{.*}}reassoc{{.*}}float @llvm.ripple.reduce.fminimum.f32(i64 1, float
// CHECKRIPPLEGEN-COUNT-2: call reassoc float @llvm.vector.reduce.fminimum.v32f32(

gen_reduce_test (16, minimum, double, f64)
// CHECKALL: check_reduction_minimum_f64
// CHECKCLANGGEN-COUNT-2: call{{.*}}reassoc{{.*}}double @llvm.ripple.reduce.fminimum.f64(i64 1, double
// CHECKRIPPLEGEN-COUNT-2: call reassoc double @llvm.vector.reduce.fminimum.v16f64(

gen_reduce_test (128, min, char, c)
gen_reduce_test (128, min, sc, sc)
gen_reduce_test (128, min, uc, uc)
gen_reduce_test (128, min, ss, ss)
gen_reduce_test (128, min, us, us)
gen_reduce_test (128, min, si, si)
gen_reduce_test (128, min, ui, ui)
gen_reduce_test (128, min, sl, sl)
gen_reduce_test (128, min, ul, ul)
gen_reduce_test (128, min, sll, sll)
gen_reduce_test (128, min, ull, ull)

// }}}

// {{{ AND

gen_reduce_test (128, and, uint8_t, u8)
// CHECKALL: check_reduction_and_u8
// CHECKCLANGGEN-COUNT-2: @llvm.ripple.reduce.and.i8(i64 1, i8
// CHECKRIPPLEGEN-COUNT-2: call i8 @llvm.vector.reduce.and.v128i8(

gen_reduce_test (128, and, int8_t, i8)
// CHECKALL: check_reduction_and_i8
// CHECKCLANGGEN-COUNT-2: @llvm.ripple.reduce.and.i8(i64 1, i8
// CHECKRIPPLEGEN-COUNT-2: call i8 @llvm.vector.reduce.and.v128i8(

gen_reduce_test (64, and, uint16_t, u16)
// CHECKALL: check_reduction_and_u16
// CHECKCLANGGEN-COUNT-2: @llvm.ripple.reduce.and.i16(i64 1, i16
// CHECKRIPPLEGEN-COUNT-2: call i16 @llvm.vector.reduce.and.v64i16(

gen_reduce_test (64, and, int16_t, i16)
// CHECKALL: check_reduction_and_i16
// CHECKCLANGGEN-COUNT-2: @llvm.ripple.reduce.and.i16(i64 1, i16
// CHECKRIPPLEGEN-COUNT-2: call i16 @llvm.vector.reduce.and.v64i16(

gen_reduce_test (32, and, uint32_t, u32)
// CHECKALL: check_reduction_and_u32
// CHECKCLANGGEN-COUNT-2: @llvm.ripple.reduce.and.i32(i64 1, i32
// CHECKRIPPLEGEN-COUNT-2: call i32 @llvm.vector.reduce.and.v32i32(

gen_reduce_test (32, and, int32_t, i32)
// CHECKALL: check_reduction_and_i32
// CHECKCLANGGEN-COUNT-2: @llvm.ripple.reduce.and.i32(i64 1, i32
// CHECKRIPPLEGEN-COUNT-2: call i32 @llvm.vector.reduce.and.v32i32(

gen_reduce_test (16, and, uint64_t, u64)
// CHECKALL: check_reduction_and_u64
// CHECKCLANGGEN-COUNT-2: @llvm.ripple.reduce.and.i64(i64 1, i64
// CHECKRIPPLEGEN-COUNT-2: call i64 @llvm.vector.reduce.and.v16i64(

gen_reduce_test (16, and, int64_t, i64)
// CHECKALL: check_reduction_and_i64
// CHECKCLANGGEN-COUNT-2: @llvm.ripple.reduce.and.i64(i64 1, i64
// CHECKRIPPLEGEN-COUNT-2: call i64 @llvm.vector.reduce.and.v16i64(

gen_reduce_test (128, and, char, c)
gen_reduce_test (128, and, sc, sc)
gen_reduce_test (128, and, uc, uc)
gen_reduce_test (128, and, ss, ss)
gen_reduce_test (128, and, us, us)
gen_reduce_test (128, and, si, si)
gen_reduce_test (128, and, ui, ui)
gen_reduce_test (128, and, sl, sl)
gen_reduce_test (128, and, ul, ul)
gen_reduce_test (128, and, sll, sll)
gen_reduce_test (128, and, ull, ull)

// }}}

// {{{ OR

gen_reduce_test (128, or, uint8_t, u8)
// CHECKALL: check_reduction_or_u8
// CHECKCLANGGEN-COUNT-2: @llvm.ripple.reduce.or.i8(i64 1, i8
// CHECKRIPPLEGEN-COUNT-2: call i8 @llvm.vector.reduce.or.v128i8(

gen_reduce_test (128, or, int8_t, i8)
// CHECKALL: check_reduction_or_i8
// CHECKCLANGGEN-COUNT-2: @llvm.ripple.reduce.or.i8(i64 1, i8
// CHECKRIPPLEGEN-COUNT-2: call i8 @llvm.vector.reduce.or.v128i8(

gen_reduce_test (64, or, uint16_t, u16)
// CHECKALL: check_reduction_or_u16
// CHECKCLANGGEN-COUNT-2: @llvm.ripple.reduce.or.i16(i64 1, i16
// CHECKRIPPLEGEN-COUNT-2: call i16 @llvm.vector.reduce.or.v64i16(

gen_reduce_test (64, or, int16_t, i16)
// CHECKALL: check_reduction_or_i16
// CHECKCLANGGEN-COUNT-2: @llvm.ripple.reduce.or.i16(i64 1, i16
// CHECKRIPPLEGEN-COUNT-2: call i16 @llvm.vector.reduce.or.v64i16(

gen_reduce_test (32, or, uint32_t, u32)
// CHECKALL: check_reduction_or_u32
// CHECKCLANGGEN-COUNT-2: @llvm.ripple.reduce.or.i32(i64 1, i32
// CHECKRIPPLEGEN-COUNT-2: call i32 @llvm.vector.reduce.or.v32i32(

gen_reduce_test (32, or, int32_t, i32)
// CHECKALL: check_reduction_or_i32
// CHECKCLANGGEN-COUNT-2: @llvm.ripple.reduce.or.i32(i64 1, i32
// CHECKRIPPLEGEN-COUNT-2: call i32 @llvm.vector.reduce.or.v32i32(

gen_reduce_test (16, or, uint64_t, u64)
// CHECKALL: check_reduction_or_u64
// CHECKCLANGGEN-COUNT-2: @llvm.ripple.reduce.or.i64(i64 1, i64
// CHECKRIPPLEGEN-COUNT-2: call i64 @llvm.vector.reduce.or.v16i64(

gen_reduce_test (16, or, int64_t, i64)
// CHECKALL: check_reduction_or_i64
// CHECKCLANGGEN-COUNT-2: @llvm.ripple.reduce.or.i64(i64 1, i64
// CHECKRIPPLEGEN-COUNT-2: call i64 @llvm.vector.reduce.or.v16i64(

gen_reduce_test (128, or, char, c)
gen_reduce_test (128, or, sc, sc)
gen_reduce_test (128, or, uc, uc)
gen_reduce_test (128, or, ss, ss)
gen_reduce_test (128, or, us, us)
gen_reduce_test (128, or, si, si)
gen_reduce_test (128, or, ui, ui)
gen_reduce_test (128, or, sl, sl)
gen_reduce_test (128, or, ul, ul)
gen_reduce_test (128, or, sll, sll)
gen_reduce_test (128, or, ull, ull)

// }}}

#ifdef __cpluplus
}
#endif
