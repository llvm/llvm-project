// REQUIRES: hexagon-registered-target
// RUN: %clang++ -g -S -fenable-ripple --target=hexagon -mhvx -mv79 -emit-llvm %s -o - -mllvm -ripple-disable-link 2>&1 | FileCheck %s

#include <ripple.h>
#include <ripple_math.h>
#include <ripple/HVX_Vdeal.h>

#define HVX_128_i8 256
#define HVX_64_i16 128
#define HVX_32_i32 64
#define HVX_16_i64 32

#define HVX_128_u8 256
#define HVX_64_u16 128
#define HVX_32_u32 64
#define HVX_16_u64 32

#define HVX_16_f64 32
#define HVX_32_f32 64
#define HVX_64_bf16 128
#define HVX_64_f16 128

#define __gen_vdeal_test(CT, T, N)                                             \
    void Ripple_vdeal_##T(size_t length, CT *dest,                             \
        CT *src, size_t chunk_size) {                                          \
      ripple_block_t BS = ripple_set_block_shape(0, HVX_##N##_##T);            \
      int v = ripple_id(BS, 0);                                                \
      src[v] = src[v] + src[v];                                                \
      dest[v] = hvx_vdeal(src[v], chunk_size);                                 \
    }

#define __gen_vshuff_test(CT, T, N)                                            \
    void Ripple_vshuff_##T(size_t length, CT *dest,                            \
        CT *src, size_t chunk_size) {                                          \
      ripple_block_t BS = ripple_set_block_shape(0, HVX_##N##_##T);            \
      int v = ripple_id(BS, 0);                                                \
      src[v] = src[v] + src[v];                                                \
      dest[v] = hvx_vshuff(src[v], chunk_size);                                \
    }

extern "C" {

// __________________________________ vdeal ____________________________________

__gen_vdeal_test(int8_t, i8, 128);
// CHECK: @Ripple_vdeal_i8
// CHECK: call <256 x i8> @ripple_pure_hvx_vdeal_i8

__gen_vdeal_test(uint8_t, u8, 128);
// CHECK: @Ripple_vdeal_u8
// CHECK: call <256 x i8> @ripple_pure_hvx_vdeal_u8

__gen_vdeal_test(int16_t, i16, 64);
// CHECK: @Ripple_vdeal_i16
// CHECK: call <128 x i16> @ripple_pure_hvx_vdeal_i16

__gen_vdeal_test(uint16_t, u16, 64);
// CHECK: @Ripple_vdeal_u16
// CHECK: call <128 x i16> @ripple_pure_hvx_vdeal_u16

__gen_vdeal_test(int32_t, i32, 32);
// CHECK: @Ripple_vdeal_i32
// CHECK: call <64 x i32> @ripple_pure_hvx_vdeal_i32

__gen_vdeal_test(uint32_t, u32, 32);
// CHECK: @Ripple_vdeal_u32
// CHECK: call <64 x i32> @ripple_pure_hvx_vdeal_u32

__gen_vdeal_test(int64_t, i64, 16);
// CHECK: @Ripple_vdeal_i64
// CHECK: call <32 x i64> @ripple_pure_hvx_vdeal_i64

__gen_vdeal_test(uint64_t, u64, 16);
// CHECK: @Ripple_vdeal_u64
// CHECK: call <32 x i64> @ripple_pure_hvx_vdeal_u64

__gen_vdeal_test(double, f64, 16);
// CHECK: @Ripple_vdeal_f64
// CHECK: call <32 x double> @ripple_pure_hvx_vdeal_f64

__gen_vdeal_test(float, f32, 32);
// CHECK: @Ripple_vdeal_f32
// CHECK: call <64 x float> @ripple_pure_hvx_vdeal_f32

__gen_vdeal_test(_Float16, f16, 64);
// CHECK: @Ripple_vdeal_f16
// CHECK: call <128 x half> @ripple_pure_hvx_vdeal_f16

// __________________________________ vshuff ___________________________________

__gen_vshuff_test(int8_t, i8, 128);
// CHECK: @Ripple_vshuff_i8
// CHECK: call <256 x i8> @ripple_pure_hvx_vshuff_i8

__gen_vshuff_test(uint8_t, u8, 128);
// CHECK: @Ripple_vshuff_u8
// CHECK: call <256 x i8> @ripple_pure_hvx_vshuff_u8

__gen_vshuff_test(int16_t, i16, 64);
// CHECK: @Ripple_vshuff_i16
// CHECK: call <128 x i16> @ripple_pure_hvx_vshuff_i16

__gen_vshuff_test(uint16_t, u16, 64);
// CHECK: @Ripple_vshuff_u16
// CHECK: call <128 x i16> @ripple_pure_hvx_vshuff_u16

__gen_vshuff_test(int32_t, i32, 32);
// CHECK: @Ripple_vshuff_i32
// CHECK: call <64 x i32> @ripple_pure_hvx_vshuff_i32

__gen_vshuff_test(uint32_t, u32, 32);
// CHECK: @Ripple_vshuff_u32
// CHECK: call <64 x i32> @ripple_pure_hvx_vshuff_u32

__gen_vshuff_test(int64_t, i64, 16);
// CHECK: @Ripple_vshuff_i64
// CHECK: call <32 x i64> @ripple_pure_hvx_vshuff_i64

__gen_vshuff_test(uint64_t, u64, 16);
// CHECK: @Ripple_vshuff_u64
// CHECK: call <32 x i64> @ripple_pure_hvx_vshuff_u64

__gen_vshuff_test(double, f64, 16);
// CHECK: @Ripple_vshuff_f64
// CHECK: call <32 x double> @ripple_pure_hvx_vshuff_f64

__gen_vshuff_test(float, f32, 32);
// CHECK: @Ripple_vshuff_f32
// CHECK: call <64 x float> @ripple_pure_hvx_vshuff_f32

__gen_vshuff_test(_Float16, f16, 64);
// CHECK: @Ripple_vshuff_f16
// CHECK: call <128 x half> @ripple_pure_hvx_vshuff_f16

}
