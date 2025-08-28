// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clangxx -S -O1 -emit-llvm %s -o - 2>&1 | FileCheck %s --implicit-check-not="warning:" --implicit-check-not="error:"

#include <ripple.h>

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

extern "C" size_t get_src_id(size_t id, size_t n) {
  return (id % 2) ? (id - 1) : (id + 1);
}

// {{{ Function pointer for index mapping test

#define gen_shuffle_fptr_test(N, LONGTYPE)                                     \
  extern "C" void check_shuffle_fptr_##LONGTYPE(LONGTYPE in[N],                \
                                                LONGTYPE out[N]) {             \
    ripple_block_t BS = ripple_set_block_shape(0, N);                                               \
    int id = ripple_id(BS, 0);                                                  \
    LONGTYPE tmp = in[id];                                                     \
    LONGTYPE shuf_tmp = ripple_shuffle(tmp, get_src_id);                       \
    out[id] = shuf_tmp;                                                        \
  }                                                                            \
  extern "C" void check_shuffle_pair_fptr_##LONGTYPE(                           \
      LONGTYPE in[N], LONGTYPE in2[N], LONGTYPE out[N]) {                      \
    ripple_block_t BS = ripple_set_block_shape(0, N);                                               \
    int id = ripple_id(BS, 0);                                                  \
    LONGTYPE tmp = in[id];                                                     \
    LONGTYPE tmp2 = in2[id];                                                   \
    LONGTYPE shuf_tmp = ripple_shuffle_pair(tmp, tmp2, get_src_id);             \
    out[id] = shuf_tmp;                                                        \
  }

gen_shuffle_fptr_test (128, uint8_t);
// CHECK: @check_shuffle_fptr_uint8_t
// CHECK: llvm.ripple.ishuffle.i8(i8 %{{[0-9]+}}, i8 %{{[0-9]+}}, i1 false, ptr nonnull @get_src_id)
// CHECK: ret
// CHECK: @check_shuffle_pair_fptr_uint8_t
// CHECK: llvm.ripple.ishuffle.i8(i8 %{{[0-9]+}}, i8 %{{[0-9]+}}, i1 true, ptr nonnull @get_src_id)
// CHECK: ret

gen_shuffle_fptr_test (128, int8_t);
// CHECK: @check_shuffle_fptr_int8_t
// CHECK: llvm.ripple.ishuffle.i8(i8 %{{[0-9]+}}, i8 %{{[0-9]+}}, i1 false, ptr nonnull @get_src_id)
// CHECK: ret
// CHECK: @check_shuffle_pair_fptr_int8_t
// CHECK: llvm.ripple.ishuffle.i8(i8 %{{[0-9]+}}, i8 %{{[0-9]+}}, i1 true, ptr nonnull @get_src_id)
// CHECK: ret

gen_shuffle_fptr_test (64, uint16_t);
// CHECK: @check_shuffle_fptr_uint16_t
// CHECK: llvm.ripple.ishuffle.i16(i16 %{{[0-9]+}}, i16 %{{[0-9]+}}, i1 false, ptr nonnull @get_src_id)
// CHECK: ret
// CHECK: @check_shuffle_pair_fptr_uint16_t
// CHECK: llvm.ripple.ishuffle.i16(i16 %{{[0-9]+}}, i16 %{{[0-9]+}}, i1 true, ptr nonnull @get_src_id)
// CHECK: ret

gen_shuffle_fptr_test (64, int16_t);
// CHECK: @check_shuffle_fptr_int16_t
// CHECK: llvm.ripple.ishuffle.i16(i16 %{{[0-9]+}}, i16 %{{[0-9]+}}, i1 false, ptr nonnull @get_src_id)
// CHECK: ret
// CHECK: @check_shuffle_pair_fptr_int16_t
// CHECK: llvm.ripple.ishuffle.i16(i16 %{{[0-9]+}}, i16 %{{[0-9]+}}, i1 true, ptr nonnull @get_src_id)
// CHECK: ret

gen_shuffle_fptr_test (32, uint32_t);
// CHECK: @check_shuffle_fptr_uint32_t
// CHECK: llvm.ripple.ishuffle.i32(i32 %{{[0-9]+}}, i32 %{{[0-9]+}}, i1 false, ptr nonnull @get_src_id)
// CHECK: ret
// CHECK: @check_shuffle_pair_fptr_uint32_t
// CHECK: llvm.ripple.ishuffle.i32(i32 %{{[0-9]+}}, i32 %{{[0-9]+}}, i1 true, ptr nonnull @get_src_id)
// CHECK: ret

gen_shuffle_fptr_test (32, int32_t);
// CHECK: @check_shuffle_fptr_int32_t
// CHECK: llvm.ripple.ishuffle.i32(i32 %{{[0-9]+}}, i32 %{{[0-9]+}}, i1 false, ptr nonnull @get_src_id)
// CHECK: ret
// CHECK: @check_shuffle_pair_fptr_int32_t
// CHECK: llvm.ripple.ishuffle.i32(i32 %{{[0-9]+}}, i32 %{{[0-9]+}}, i1 true, ptr nonnull @get_src_id)
// CHECK: ret

gen_shuffle_fptr_test (16, uint64_t);
// CHECK: @check_shuffle_fptr_uint64_t
// CHECK: llvm.ripple.ishuffle.i64(i64 %{{[0-9]+}}, i64 %{{[0-9]+}}, i1 false, ptr nonnull @get_src_id)
// CHECK: ret
// CHECK: @check_shuffle_pair_fptr_uint64_t
// CHECK: llvm.ripple.ishuffle.i64(i64 %{{[0-9]+}}, i64 %{{[0-9]+}}, i1 true, ptr nonnull @get_src_id)
// CHECK: ret

gen_shuffle_fptr_test (16, int64_t);
// CHECK: @check_shuffle_fptr_int64_t
// CHECK: llvm.ripple.ishuffle.i64(i64 %{{[0-9]+}}, i64 %{{[0-9]+}}, i1 false, ptr nonnull @get_src_id)
// CHECK: ret
// CHECK: @check_shuffle_pair_fptr_int64_t
// CHECK: llvm.ripple.ishuffle.i64(i64 %{{[0-9]+}}, i64 %{{[0-9]+}}, i1 true, ptr nonnull @get_src_id)
// CHECK: ret

gen_shuffle_fptr_test(16, float);
// CHECK: @check_shuffle_fptr_float
// CHECK: llvm.ripple.fshuffle.f32(float %{{[0-9]+}}, float %{{[0-9]+}}, i1 false, ptr nonnull @get_src_id)
// CHECK: ret
// CHECK: @check_shuffle_pair_fptr_float
// CHECK: llvm.ripple.fshuffle.f32(float %{{[0-9]+}}, float %{{[0-9]+}}, i1 true, ptr nonnull @get_src_id)
// CHECK: ret

gen_shuffle_fptr_test(16, double);
// CHECK: @check_shuffle_fptr_double
// CHECK: llvm.ripple.fshuffle.f64(double %{{[0-9]+}}, double %{{[0-9]+}}, i1 false, ptr nonnull @get_src_id)
// CHECK: ret
// CHECK: @check_shuffle_pair_fptr_double
// CHECK: llvm.ripple.fshuffle.f64(double %{{[0-9]+}}, double %{{[0-9]+}}, i1 true, ptr nonnull @get_src_id)
// CHECK: ret

gen_shuffle_fptr_test(128, char);
gen_shuffle_fptr_test(128, sc);
gen_shuffle_fptr_test(128, uc);
gen_shuffle_fptr_test(128, ss);
gen_shuffle_fptr_test(128, us);
gen_shuffle_fptr_test(128, si);
gen_shuffle_fptr_test(128, ui);
gen_shuffle_fptr_test(128, sl);
gen_shuffle_fptr_test(128, ul);
gen_shuffle_fptr_test(128, sll);
gen_shuffle_fptr_test(128, ull);

// }}}


// {{{ Lambda decl for index mapping test

#define gen_shuffle_lambda_test(N, LONGTYPE)                                   \
  extern "C" void check_shuffle_lambda_##LONGTYPE(LONGTYPE in[N],              \
                                                  LONGTYPE out[N]) {           \
    ripple_block_t BS = ripple_set_block_shape(0, N);                                               \
    int id = ripple_id(BS, 0);                                                  \
    LONGTYPE tmp = in[id];                                                     \
    LONGTYPE shuf_tmp = ripple_shuffle(                                        \
        tmp, [](size_t i, size_t n) { return (i % 2) ? (i - 1) : (i + 1); });  \
    out[id] = shuf_tmp;                                                        \
  }                                                                            \
  extern "C" void check_shuffle_pair_lambda_##LONGTYPE(                         \
      LONGTYPE in[N], LONGTYPE in2[N], LONGTYPE out[N]) {                      \
    ripple_block_t BS = ripple_set_block_shape(0, N);                                               \
    int id = ripple_id(BS, 0);                                                  \
    LONGTYPE tmp = in[id];                                                     \
    LONGTYPE tmp2 = in2[id];                                                   \
    LONGTYPE shuf_tmp = ripple_shuffle_pair(tmp, tmp2, [](size_t i, size_t n) { \
      return (i % 2) ? n + (i - 1) : (i + 1);                                  \
    });                                                                        \
    out[id] = shuf_tmp;                                                        \
  }

gen_shuffle_lambda_test (128, uint8_t);
// CHECK: @check_shuffle_lambda_uint8_t
// CHECK: llvm.ripple.ishuffle.i8(i8 %{{[0-9]+}}, i8 %{{[0-9]+}}, i1 false, ptr nonnull{{.*}})
// CHECK: ret
// CHECK: @check_shuffle_pair_lambda_uint8_t
// CHECK: llvm.ripple.ishuffle.i8(i8 %{{[0-9]+}}, i8 %{{[0-9]+}}, i1 true, ptr nonnull{{.*}})
// CHECK: ret

gen_shuffle_lambda_test (128, int8_t);
// CHECK: @check_shuffle_lambda_int8_t
// CHECK: llvm.ripple.ishuffle.i8(i8 %{{[0-9]+}}, i8 %{{[0-9]+}}, i1 false, ptr nonnull{{.*}})
// CHECK: ret
// CHECK: @check_shuffle_pair_lambda_int8_t
// CHECK: llvm.ripple.ishuffle.i8(i8 %{{[0-9]+}}, i8 %{{[0-9]+}}, i1 true, ptr nonnull{{.*}})
// CHECK: ret

gen_shuffle_lambda_test (64, uint16_t);
// CHECK: @check_shuffle_lambda_uint16_t
// CHECK: llvm.ripple.ishuffle.i16(i16 %{{[0-9]+}}, i16 %{{[0-9]+}}, i1 false, ptr nonnull{{.*}})
// CHECK: ret
// CHECK: @check_shuffle_pair_lambda_uint16_t
// CHECK: llvm.ripple.ishuffle.i16(i16 %{{[0-9]+}}, i16 %{{[0-9]+}}, i1 true, ptr nonnull{{.*}})
// CHECK: ret

gen_shuffle_lambda_test (64, int16_t);
// CHECK: @check_shuffle_lambda_int16_t
// CHECK: llvm.ripple.ishuffle.i16(i16 %{{[0-9]+}}, i16 %{{[0-9]+}}, i1 false, ptr nonnull{{.*}})
// CHECK: ret
// CHECK: @check_shuffle_pair_lambda_int16_t
// CHECK: llvm.ripple.ishuffle.i16(i16 %{{[0-9]+}}, i16 %{{[0-9]+}}, i1 true, ptr nonnull{{.*}})
// CHECK: ret


gen_shuffle_lambda_test (32, uint32_t);
// CHECK: @check_shuffle_lambda_uint32_t
// CHECK: llvm.ripple.ishuffle.i32(i32 %{{[0-9]+}}, i32 %{{[0-9]+}}, i1 false, ptr nonnull{{.*}})
// CHECK: ret
// CHECK: @check_shuffle_pair_lambda_uint32_t
// CHECK: llvm.ripple.ishuffle.i32(i32 %{{[0-9]+}}, i32 %{{[0-9]+}}, i1 true, ptr nonnull{{.*}})
// CHECK: ret

gen_shuffle_lambda_test (32, int32_t);
// CHECK: @check_shuffle_lambda_int32_t
// CHECK: llvm.ripple.ishuffle.i32(i32 %{{[0-9]+}}, i32 %{{[0-9]+}}, i1 false, ptr nonnull{{.*}})
// CHECK: ret
// CHECK: @check_shuffle_pair_lambda_int32_t
// CHECK: llvm.ripple.ishuffle.i32(i32 %{{[0-9]+}}, i32 %{{[0-9]+}}, i1 true, ptr nonnull{{.*}})
// CHECK: ret

gen_shuffle_lambda_test (16, uint64_t);
// CHECK: @check_shuffle_lambda_uint64_t
// CHECK: llvm.ripple.ishuffle.i64(i64 %{{[0-9]+}}, i64 %{{[0-9]+}}, i1 false, ptr nonnull{{.*}})
// CHECK: ret
// CHECK: @check_shuffle_pair_lambda_uint64_t
// CHECK: llvm.ripple.ishuffle.i64(i64 %{{[0-9]+}}, i64 %{{[0-9]+}}, i1 true, ptr nonnull{{.*}})
// CHECK: ret

gen_shuffle_lambda_test (16, int64_t);
// CHECK: @check_shuffle_lambda_int64_t
// CHECK: llvm.ripple.ishuffle.i64(i64 %{{[0-9]+}}, i64 %{{[0-9]+}}, i1 false, ptr nonnull{{.*}})
// CHECK: ret
// CHECK: @check_shuffle_pair_lambda_int64_t
// CHECK: llvm.ripple.ishuffle.i64(i64 %{{[0-9]+}}, i64 %{{[0-9]+}}, i1 true, ptr nonnull{{.*}})
// CHECK: ret

gen_shuffle_lambda_test(16, float);
// CHECK: @check_shuffle_lambda_float
// CHECK: llvm.ripple.fshuffle.f32(float %{{[0-9]+}}, float %{{[0-9]+}}, i1 false, ptr nonnull{{.*}})
// CHECK: ret
// CHECK: @check_shuffle_pair_lambda_float
// CHECK: llvm.ripple.fshuffle.f32(float %{{[0-9]+}}, float %{{[0-9]+}}, i1 true, ptr nonnull{{.*}})
// CHECK: ret

gen_shuffle_lambda_test(16, double);
// CHECK: @check_shuffle_lambda_double
// CHECK: llvm.ripple.fshuffle.f64(double %{{[0-9]+}}, double %{{[0-9]+}}, i1 false, ptr nonnull{{.*}})
// CHECK: ret
// CHECK: @check_shuffle_pair_lambda_double
// CHECK: llvm.ripple.fshuffle.f64(double %{{[0-9]+}}, double %{{[0-9]+}}, i1 true, ptr nonnull{{.*}})
// CHECK: ret

gen_shuffle_lambda_test(128, char);
gen_shuffle_lambda_test(128, sc);
gen_shuffle_lambda_test(128, uc);
gen_shuffle_lambda_test(128, ss);
gen_shuffle_lambda_test(128, us);
gen_shuffle_lambda_test(128, si);
gen_shuffle_lambda_test(128, ui);
gen_shuffle_lambda_test(128, sl);
gen_shuffle_lambda_test(128, ul);
gen_shuffle_lambda_test(128, sll);
gen_shuffle_lambda_test(128, ull);

// }}}


// {{{ Lambda ref for index mapping test

#define gen_shuffle_lambda_ref_test(N, LONGTYPE)                               \
  extern "C" void check_shuffle_lambda_ref_##LONGTYPE(LONGTYPE in[N],          \
                                                      LONGTYPE out[N]) {       \
    ripple_block_t BS = ripple_set_block_shape(0, N);                          \
    int id = ripple_id(BS, 0);                                                 \
    LONGTYPE tmp = in[id];                                                     \
    auto idx_map = [](size_t i, size_t n) {                                    \
      return (i % 2) ? (i - 1) : (i + 1);                                      \
    };                                                                         \
    LONGTYPE shuf_tmp = ripple_shuffle(tmp, idx_map);                          \
    out[id] = shuf_tmp;                                                        \
  }                                                                            \
  extern "C" void check_shuffle_pair_lambda_ref_##LONGTYPE(                    \
      LONGTYPE in[N], LONGTYPE in2[N], LONGTYPE out[N]) {                      \
    ripple_block_t BS = ripple_set_block_shape(0, N);                          \
    int id = ripple_id(BS, 0);                                                 \
    LONGTYPE tmp = in[id];                                                     \
    LONGTYPE tmp2 = in2[id];                                                   \
    auto idx_map = [](size_t i, size_t n) {                                    \
      return (i % 2) ? (i - 1) : (i + 1);                                      \
    };                                                                         \
    LONGTYPE shuf_tmp = ripple_shuffle_pair(tmp, tmp2, idx_map);               \
    out[id] = shuf_tmp;                                                        \
  }

gen_shuffle_lambda_ref_test (128, uint8_t);
// CHECK: @check_shuffle_lambda_ref_uint8_t
// CHECK: llvm.ripple.ishuffle.i8(i8 %{{[0-9]+}}, i8 %{{[0-9]+}}, i1 false, ptr nonnull{{.*}})
// CHECK: ret
// CHECK: @check_shuffle_pair_lambda_ref_uint8_t
// CHECK: llvm.ripple.ishuffle.i8(i8 %{{[0-9]+}}, i8 %{{[0-9]+}}, i1 true, ptr nonnull{{.*}})
// CHECK: ret

gen_shuffle_lambda_ref_test (128, int8_t);
// CHECK: @check_shuffle_lambda_ref_int8_t
// CHECK: llvm.ripple.ishuffle.i8(i8 %{{[0-9]+}}, i8 %{{[0-9]+}}, i1 false, ptr nonnull{{.*}})
// CHECK: ret
// CHECK: @check_shuffle_pair_lambda_ref_int8_t
// CHECK: llvm.ripple.ishuffle.i8(i8 %{{[0-9]+}}, i8 %{{[0-9]+}}, i1 true, ptr nonnull{{.*}})
// CHECK: ret

gen_shuffle_lambda_ref_test (64, uint16_t);
// CHECK: @check_shuffle_lambda_ref_uint16_t
// CHECK: llvm.ripple.ishuffle.i16(i16 %{{[0-9]+}}, i16 %{{[0-9]+}}, i1 false, ptr nonnull{{.*}})
// CHECK: ret
// CHECK: @check_shuffle_pair_lambda_ref_uint16_t
// CHECK: llvm.ripple.ishuffle.i16(i16 %{{[0-9]+}}, i16 %{{[0-9]+}}, i1 true, ptr nonnull{{.*}})
// CHECK: ret

gen_shuffle_lambda_ref_test (64, int16_t);
// CHECK: @check_shuffle_lambda_ref_int16_t
// CHECK: llvm.ripple.ishuffle.i16(i16 %{{[0-9]+}}, i16 %{{[0-9]+}}, i1 false, ptr nonnull{{.*}})
// CHECK: ret
// CHECK: @check_shuffle_pair_lambda_ref_int16_t
// CHECK: llvm.ripple.ishuffle.i16(i16 %{{[0-9]+}}, i16 %{{[0-9]+}}, i1 true, ptr nonnull{{.*}})
// CHECK: ret

gen_shuffle_lambda_ref_test (32, uint32_t);
// CHECK: @check_shuffle_lambda_ref_uint32_t
// CHECK: llvm.ripple.ishuffle.i32(i32 %{{[0-9]+}}, i32 %{{[0-9]+}}, i1 false, ptr nonnull{{.*}})
// CHECK: ret
// CHECK: @check_shuffle_pair_lambda_ref_uint32_t
// CHECK: llvm.ripple.ishuffle.i32(i32 %{{[0-9]+}}, i32 %{{[0-9]+}}, i1 true, ptr nonnull{{.*}})
// CHECK: ret

gen_shuffle_lambda_ref_test (32, int32_t);
// CHECK: @check_shuffle_lambda_ref_int32_t
// CHECK: llvm.ripple.ishuffle.i32(i32 %{{[0-9]+}}, i32 %{{[0-9]+}}, i1 false, ptr nonnull{{.*}})
// CHECK: ret
// CHECK: @check_shuffle_pair_lambda_ref_int32_t
// CHECK: llvm.ripple.ishuffle.i32(i32 %{{[0-9]+}}, i32 %{{[0-9]+}}, i1 true, ptr nonnull{{.*}})
// CHECK: ret

gen_shuffle_lambda_ref_test (16, uint64_t);
// CHECK: @check_shuffle_lambda_ref_uint64_t
// CHECK: llvm.ripple.ishuffle.i64(i64 %{{[0-9]+}}, i64 %{{[0-9]+}}, i1 false, ptr nonnull{{.*}})
// CHECK: ret
// CHECK: @check_shuffle_pair_lambda_ref_uint64_t
// CHECK: llvm.ripple.ishuffle.i64(i64 %{{[0-9]+}}, i64 %{{[0-9]+}}, i1 true, ptr nonnull{{.*}})
// CHECK: ret

gen_shuffle_lambda_ref_test (16, int64_t);
// CHECK: @check_shuffle_lambda_ref_int64_t
// CHECK: llvm.ripple.ishuffle.i64(i64 %{{[0-9]+}}, i64 %{{[0-9]+}}, i1 false, ptr nonnull{{.*}})
// CHECK: ret
// CHECK: @check_shuffle_pair_lambda_ref_int64_t
// CHECK: llvm.ripple.ishuffle.i64(i64 %{{[0-9]+}}, i64 %{{[0-9]+}}, i1 true, ptr nonnull{{.*}})
// CHECK: ret

gen_shuffle_lambda_ref_test (16, float);
// CHECK: @check_shuffle_lambda_ref_float
// CHECK: llvm.ripple.fshuffle.f32(float %{{[0-9]+}}, float %{{[0-9]+}}, i1 false, ptr nonnull{{.*}})
// CHECK: ret
// CHECK: @check_shuffle_pair_lambda_ref_float
// CHECK: llvm.ripple.fshuffle.f32(float %{{[0-9]+}}, float %{{[0-9]+}}, i1 true, ptr nonnull{{.*}})
// CHECK: ret

gen_shuffle_lambda_ref_test (16, double);
// CHECK: @check_shuffle_lambda_ref_double
// CHECK: llvm.ripple.fshuffle.f64(double %{{[0-9]+}}, double %{{[0-9]+}}, i1 false, ptr nonnull{{.*}})
// CHECK: ret
// CHECK: @check_shuffle_pair_lambda_ref_double
// CHECK: llvm.ripple.fshuffle.f64(double %{{[0-9]+}}, double %{{[0-9]+}}, i1 true, ptr nonnull{{.*}})
// CHECK: ret

gen_shuffle_lambda_ref_test(128, char);
gen_shuffle_lambda_ref_test(128, sc);
gen_shuffle_lambda_ref_test(128, uc);
gen_shuffle_lambda_ref_test(128, ss);
gen_shuffle_lambda_ref_test(128, us);
gen_shuffle_lambda_ref_test(128, si);
gen_shuffle_lambda_ref_test(128, ui);
gen_shuffle_lambda_ref_test(128, sl);
gen_shuffle_lambda_ref_test(128, ul);
gen_shuffle_lambda_ref_test(128, sll);
gen_shuffle_lambda_ref_test(128, ull);

// }}}
