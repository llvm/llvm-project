// UNSUPPORTED: cui
// REQUIRES: hexagon-registered-target
// REQUIRES: rt_lib
// RUN: %clang++ -g -S -fenable-ripple --target=hexagon -mhvx -mv79 -emit-llvm %s -o - -mllvm -ripple-disable-link 2>&1 | FileCheck %s

#include <ripple.h>
#include <ripple_math.h>
#include <ripple/HVX_Splice.h>

#define __gen_splice_test(CT, T, N)                                             \
    void Ripple_splice_##T(size_t length, CT *dest,                             \
        CT *left, CT *right, size_t start) {                                   \
      ripple_block_t BS = ripple_set_block_shape(0, N);                        \
      int v = ripple_id(BS, 0);                                                \
      dest[v] = hvx_splice(left[v], right[v], start);                           \
    }

#define __gen_lsplice_test(CT, T, N)                                            \
    void Ripple_lsplice_##T(size_t length, CT *dest,                            \
        CT *left, CT *right, size_t start) {                                   \
      ripple_block_t BS = ripple_set_block_shape(0, N);                        \
      int v = ripple_id(BS, 0);                                                \
      dest[v] = hvx_lsplice(left[v], right[v], start);                          \
    }

extern "C" {

// __________________________________ splice ____________________________________

__gen_splice_test(int8_t, i8, 128);
// CHECK: @Ripple_splice_i8
// CHECK: call <128 x i8> @ripple_pure_hvx_splice_i8

__gen_splice_test(uint8_t, u8, 128);
// CHECK: @Ripple_splice_u8
// CHECK: call <128 x i8> @ripple_pure_hvx_splice_u8

__gen_splice_test(int16_t, i16, 64);
// CHECK: @Ripple_splice_i16
// CHECK: call <64 x i16> @ripple_pure_hvx_splice_i16

__gen_splice_test(uint16_t, u16, 64);
// CHECK: @Ripple_splice_u16
// CHECK: call <64 x i16> @ripple_pure_hvx_splice_u16

__gen_splice_test(int32_t, i32, 32);
// CHECK: @Ripple_splice_i32
// CHECK: call <32 x i32> @ripple_pure_hvx_splice_i32

__gen_splice_test(uint32_t, u32, 32);
// CHECK: @Ripple_splice_u32
// CHECK: call <32 x i32> @ripple_pure_hvx_splice_u32

__gen_splice_test(int64_t, i64, 16);
// CHECK: @Ripple_splice_i64
// CHECK: call <16 x i64> @ripple_pure_hvx_splice_i64

__gen_splice_test(uint64_t, u64, 16);
// CHECK: @Ripple_splice_u64
// CHECK: call <16 x i64> @ripple_pure_hvx_splice_u64

__gen_splice_test(double, f64, 16);
// CHECK: @Ripple_splice_f64
// CHECK: call <16 x double> @ripple_pure_hvx_splice_f64

__gen_splice_test(float, f32, 32);
// CHECK: @Ripple_splice_f32
// CHECK: call <32 x float> @ripple_pure_hvx_splice_f32

__gen_splice_test(_Float16, f16, 64);
// CHECK: @Ripple_splice_f16
// CHECK: call <64 x half> @ripple_pure_hvx_splice_f16

// __________________________________ lsplice ___________________________________

__gen_lsplice_test(int8_t, i8, 128);
// CHECK: @Ripple_lsplice_i8
// CHECK: call <128 x i8> @ripple_pure_hvx_lsplice_i8

__gen_lsplice_test(uint8_t, u8, 128);
// CHECK: @Ripple_lsplice_u8
// CHECK: call <128 x i8> @ripple_pure_hvx_lsplice_u8

__gen_lsplice_test(int16_t, i16, 64);
// CHECK: @Ripple_lsplice_i16
// CHECK: call <64 x i16> @ripple_pure_hvx_lsplice_i16

__gen_lsplice_test(uint16_t, u16, 64);
// CHECK: @Ripple_lsplice_u16
// CHECK: call <64 x i16> @ripple_pure_hvx_lsplice_u16

__gen_lsplice_test(int32_t, i32, 32);
// CHECK: @Ripple_lsplice_i32
// CHECK: call <32 x i32> @ripple_pure_hvx_lsplice_i32

__gen_lsplice_test(uint32_t, u32, 32);
// CHECK: @Ripple_lsplice_u32
// CHECK: call <32 x i32> @ripple_pure_hvx_lsplice_u32

__gen_lsplice_test(int64_t, i64, 16);
// CHECK: @Ripple_lsplice_i64
// CHECK: call <16 x i64> @ripple_pure_hvx_lsplice_i64

__gen_lsplice_test(uint64_t, u64, 16);
// CHECK: @Ripple_lsplice_u64
// CHECK: call <16 x i64> @ripple_pure_hvx_lsplice_u64

__gen_lsplice_test(double, f64, 16);
// CHECK: @Ripple_lsplice_f64
// CHECK: call <16 x double> @ripple_pure_hvx_lsplice_f64

__gen_lsplice_test(float, f32, 32);
// CHECK: @Ripple_lsplice_f32
// CHECK: call <32 x float> @ripple_pure_hvx_lsplice_f32

__gen_lsplice_test(_Float16, f16, 64);
// CHECK: @Ripple_lsplice_f16
// CHECK: call <64 x half> @ripple_pure_hvx_lsplice_f16

}