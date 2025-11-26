//===------- ripple_hvx.h: Hexagon Vector ripple helpers ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#if defined(__hexagon__)

// ________________________ Mock conversion from and to hvx ____________________

// Here we define a set of standard C vector types
// which correspond to the HVX native types.
// The syntax follows that of LLVN: v{n_elems}{type_letter}{elem_precision},
// where 'type_letter' is i for ints and f for floats

// This code can be compiled for other targets.
// We just need these particular sizes to match the HVX native vectors.

// Note: this only declares HVX native vector types.
// Add more __decl_* lines to declare vector types that
// that are suitable for other architectures.
#include "ripple.h"

#define __decl_f_vec_t(CT, T, W, N, BW)                                        \
  typedef CT v##N##T##W __attribute__((vector_size(BW)))                       \
  __attribute__((aligned(BW)))

#if __has_bf16__
__decl_f_vec_t(__bf16, bf, 16, 64, 128);
__decl_f_vec_t(__bf16, bf, 16, 128, 256);
__decl_f_vec_t(__bf16, bf, 16, 256, 512);
#endif

#if __has_Float16__
__decl_f_vec_t(_Float16, f, 16, 64, 128);
__decl_f_vec_t(_Float16, f, 16, 128, 256);
__decl_f_vec_t(_Float16, f, 16, 256, 512);
#endif

__decl_f_vec_t(float, f, 32, 32, 128);
__decl_f_vec_t(float, f, 32, 64, 256);
__decl_f_vec_t(float, f, 32, 128, 512);
__decl_f_vec_t(double, f, 64, 16, 128);
__decl_f_vec_t(double, f, 64, 32, 256);
__decl_f_vec_t(double, f, 64, 64, 512);

#undef __decl_f_vec_t

#define __decl_int_vec_t(EL_WIDTH, VEC_SIZE, VEC_BYTE_WIDTH)                   \
  typedef int##EL_WIDTH##_t v##VEC_SIZE##i##EL_WIDTH                           \
      __attribute__((vector_size(VEC_BYTE_WIDTH)));                            \
  typedef uint##EL_WIDTH##_t v##VEC_SIZE##u##EL_WIDTH                          \
      __attribute__((vector_size(VEC_BYTE_WIDTH)))
__decl_int_vec_t(8, 128, 128);
__decl_int_vec_t(16, 64, 128);
__decl_int_vec_t(32, 32, 128);
__decl_int_vec_t(64, 16, 128);
// Vector pairs
__decl_int_vec_t(8, 256, 256);
__decl_int_vec_t(16, 128, 256);
__decl_int_vec_t(32, 64, 256);
__decl_int_vec_t(64, 32, 256);
// Vector quads
__decl_int_vec_t(8, 512, 512);
__decl_int_vec_t(16, 256, 512);
__decl_int_vec_t(32, 128, 512);
__decl_int_vec_t(64, 64, 512);
#undef __decl_int_vec_t


union __hvx_native_vec_types {
  v128i8 i8;
  v128u8 u8;
  v64i16 i16;
  v64u16 u16;
#if __has_Float16__
  v64f16 f16;
#endif
#if __has_bf16__
  v64bf16 bf16;
#endif
  v32i32 i32;
  v32u32 u32;
  v32f32 f32;
  v16i64 i64;
  v16u64 u64;
  v16f64 f64;
};

#ifdef __cplusplus
/// @brief declares a cast from hvx to ripple and and from ripple to hvx
/// @param CT the c name of the element type
/// @param ST the "standard" name of the element type (e.g. u8, f32)
/// @param N_EL the number of elements in the (non-i32) vector
#define __decl_hvx_cast(CT, ST, N_EL, N32_EL)                                  \
  [[gnu::always_inline]] static v##N32_EL##i32 hvx_cast_to_i32(                \
      v##N_EL##ST x) {                                                         \
    v##N32_EL##i32 target;                                                     \
    static_assert(sizeof(x) == sizeof(target), "Incompatible vector sizes");   \
    __builtin_memcpy(&target, &x, sizeof(target));                             \
    return target;                                                             \
  }                                                                            \
  [[gnu::always_inline]] static v##N_EL##ST hvx_cast_from_i32(                 \
      v##N32_EL##i32 x, CT y) {                                                \
    (void)y;                                                                   \
    v##N_EL##ST target;                                                        \
    static_assert(sizeof(x) == sizeof(target), "Incompatible vector sizes");   \
    __builtin_memcpy(&target, &x, sizeof(target));                             \
    return target;                                                             \
  }

__decl_hvx_cast(int8_t, i8, 128, 32);
__decl_hvx_cast(uint8_t, u8, 128, 32);
__decl_hvx_cast(int8_t, i8, 256, 64);
__decl_hvx_cast(uint8_t, u8, 256, 64);
__decl_hvx_cast(int16_t, i16, 64, 32);
__decl_hvx_cast(uint16_t, u16, 64, 32);
__decl_hvx_cast(int16_t, i16, 128, 64);
__decl_hvx_cast(uint16_t, u16, 128, 64);
#if __has_Float16__
__decl_hvx_cast(_Float16, f16, 64, 32);
__decl_hvx_cast(_Float16, f16, 128, 64);
#endif
#if __has_bf16__
__decl_hvx_cast(__bf16, bf16, 64, 32);
__decl_hvx_cast(__bf16, bf16, 128, 64);
#endif
__decl_hvx_cast(int32_t, i32, 32, 32);
__decl_hvx_cast(uint32_t, u32, 32, 32);
__decl_hvx_cast(int32_t, i32, 64, 64);
__decl_hvx_cast(uint32_t, u32, 64, 64);
__decl_hvx_cast(float, f32, 32, 32);
__decl_hvx_cast(float, f32, 64, 64);
__decl_hvx_cast(int64_t, i64, 16, 32);
__decl_hvx_cast(uint64_t, u64, 16, 32);
__decl_hvx_cast(double, f64, 16, 32);
__decl_hvx_cast(int64_t, i64, 32, 64);
__decl_hvx_cast(uint64_t, u64, 32, 64);
__decl_hvx_cast(double, f64, 32, 64);
#undef __decl_hvx_cast

#else // __cplusplus

#define __hvx_cast_to_i32(T, x)                                                \
  ({                                                                           \
    union __hvx_native_vec_types ts;                                           \
    ts.T = (x);                                                                \
    ts.i32;                                                                    \
  })

#define __hvx_cast_from_i32(T, x)                                              \
  ({                                                                           \
    union __hvx_native_vec_types ts;                                           \
    ts.i32 = (x);                                                              \
    ts.T;                                                                      \
  })

#if __has_Float16__
#define __extra_f16_hvx_cast_to_i32(x) , v64f16 : __hvx_cast_to_i32(f16, (x))
#define __extra_f16_hvx_cast_from_i32(x)                                       \
  , _Float16 : __hvx_cast_from_i32(f16, (x))
#else
#define __extra_f16_hvx_cast_to_i32(x)
#define __extra_f16_hvx_cast_from_i32(x)
#endif

#if __has_bf16__
#define __extra_bf16_hvx_cast_to_i32(x) , v64bf16 : __hvx_cast_to_i32(bf16, (x))
#define __extra_bf16_hvx_cast_from_i32(x)                                      \
  , __bf16 : __hvx_cast_from_i32(bf16, (x))
#else
#define __extra_bf16_hvx_cast_to_i32(x)
#define __extra_bf16_hvx_cast_from_i32(x)
#endif

#define hvx_cast_to_i32(x)                                                     \
  _Generic((x),                                                                \
      v128i8: __hvx_cast_to_i32(i8, (x)),                                      \
      v128u8: __hvx_cast_to_i32(u8, (x)),                                      \
      v64i16: __hvx_cast_to_i32(i16, (x)),                                     \
      v64u16: __hvx_cast_to_i32(u16, (x)),                                     \
      v32i32: __hvx_cast_to_i32(i32, (x)),                                     \
      v32u32: __hvx_cast_to_i32(u32, (x)),                                     \
      v32f32: __hvx_cast_to_i32(f32, (x)),                                     \
      v16i64: __hvx_cast_to_i32(i64, (x)),                                     \
      v16u64: __hvx_cast_to_i32(u64, (x)),                                     \
      v16f64: __hvx_cast_to_i32(f64, (x)) __extra_bf16_hvx_cast_to_i32((x))    \
          __extra_f16_hvx_cast_to_i32((x)))

#define hvx_cast_from_i32(x, y)                                                \
  _Generic((y),                                                                \
      int8_t: __hvx_cast_from_i32(i8, (x)),                                    \
      uint8_t: __hvx_cast_from_i32(u8, (x)),                                   \
      int16_t: __hvx_cast_from_i32(i16, (x)),                                  \
      uint16_t: __hvx_cast_from_i32(u16, (x)),                                 \
      int32_t: __hvx_cast_from_i32(i32, (x)),                                  \
      uint32_t: __hvx_cast_from_i32(u32, (x)),                                 \
      float: __hvx_cast_from_i32(f32, (x)),                                    \
      int64_t: __hvx_cast_from_i32(i64, (x)),                                  \
      uint64_t: __hvx_cast_from_i32(u64, (x)),                                 \
      double: __hvx_cast_from_i32(f64, (x)) __extra_f16_hvx_cast_from_i32((x)) \
          __extra_bf16_hvx_cast_from_i32((x)))

#endif // __cplusplus

#ifdef __cplusplus

/// @brief HVX vector to Ripple representation.
/// Currently implemented for native HVX vectors only.
// The block size needs to match the vector argument size for this to work.
#define __decl_hvx_to_ripple(N_EL, C_T, T, PREC)                               \
  [[gnu::always_inline]] static C_T hvx_to_ripple_v##N_EL##T##PREC(            \
      ripple_block_t BS, v##N_EL##T##PREC x) {                                 \
    C_T tmp[N_EL];                                                             \
    *((v##N_EL##T##PREC *)tmp) = x;                                            \
    return tmp[ripple_id(BS, 0)];                                              \
  }                                                                            \
  [[gnu::always_inline]] static C_T hvx_to_ripple_2d_v##N_EL##T##PREC(         \
      ripple_block_t BS, v##N_EL##T##PREC x) {                                 \
    C_T tmp[N_EL];                                                             \
    *((v##N_EL##T##PREC *)tmp) = x;                                            \
    return tmp[ripple_id(BS, 1) * ripple_get_block_size(BS, 0) +               \
               ripple_id(BS, 0)];                                              \
  }

#else // !__cplusplus
/// @brief HVX vector to Ripple representation.
/// Currently implemented for native HVX vectors only.
// The block size needs to match the vector argument size for this to work.
#define __decl_hvx_to_ripple(N_EL, C_T, T, PREC)                               \
  __attribute__((always_inline)) static C_T hvx_to_ripple_v##N_EL##T##PREC(    \
      ripple_block_t BS, v##N_EL##T##PREC x) {                                 \
    C_T tmp[N_EL];                                                             \
    *((v##N_EL##T##PREC *)tmp) = x;                                            \
    return tmp[ripple_id(BS, 0)];                                              \
  }                                                                            \
  __attribute__((always_inline)) static C_T hvx_to_ripple_2d_v##N_EL##T##PREC( \
      ripple_block_t BS, v##N_EL##T##PREC x) {                                 \
    C_T tmp[N_EL];                                                             \
    *((v##N_EL##T##PREC *)tmp) = x;                                            \
    return tmp[ripple_id(BS, 1) * ripple_get_block_size(BS, 0) +               \
               ripple_id(BS, 0)];                                              \
  }

#endif // __cplusplus

// Declare conversions for 1d and 2-d single-, double- and quad- vectors
__decl_hvx_to_ripple(128, int8_t, i, 8);
__decl_hvx_to_ripple(128, uint8_t, u, 8);
__decl_hvx_to_ripple(256, int8_t, i, 8);
__decl_hvx_to_ripple(256, uint8_t, u, 8);
__decl_hvx_to_ripple(512, int8_t, i, 8);
__decl_hvx_to_ripple(512, uint8_t, u, 8);

__decl_hvx_to_ripple(64, int16_t, i, 16);
__decl_hvx_to_ripple(64, uint16_t, u, 16);
__decl_hvx_to_ripple(128, int16_t, i, 16);
__decl_hvx_to_ripple(128, uint16_t, u, 16);
__decl_hvx_to_ripple(256, int16_t, i, 16);
__decl_hvx_to_ripple(256, uint16_t, u, 16);

__decl_hvx_to_ripple(32, int32_t, i, 32);
__decl_hvx_to_ripple(32, uint32_t, u, 32);
__decl_hvx_to_ripple(32, float, f, 32);
__decl_hvx_to_ripple(64, int32_t, i, 32);
__decl_hvx_to_ripple(64, uint32_t, u, 32);
__decl_hvx_to_ripple(64, float, f, 32);
__decl_hvx_to_ripple(128, int32_t, i, 32);
__decl_hvx_to_ripple(128, uint32_t, u, 32);
__decl_hvx_to_ripple(128, float, f, 32);

__decl_hvx_to_ripple(16, int64_t, i, 64);
__decl_hvx_to_ripple(16, uint64_t, u, 64);
__decl_hvx_to_ripple(16, double, f, 64);
__decl_hvx_to_ripple(32, int64_t, i, 64);
__decl_hvx_to_ripple(32, uint64_t, u, 64);
__decl_hvx_to_ripple(32, double, f, 64);
__decl_hvx_to_ripple(64, int64_t, i, 64);
__decl_hvx_to_ripple(64, uint64_t, u, 64);
__decl_hvx_to_ripple(64, double, f, 64);
#if __has_Float16__
__decl_hvx_to_ripple(64, _Float16, f, 16);
__decl_hvx_to_ripple(128, _Float16, f, 16);
__decl_hvx_to_ripple(256, _Float16, f, 16);
#endif // __has_Float16__
#if __has_bf16__
__decl_hvx_to_ripple(64, __bf16, bf, 16);
__decl_hvx_to_ripple(128, __bf16, bf, 16);
__decl_hvx_to_ripple(256, __bf16, bf, 16);
// Nicer-looking macro
#undef __decl_hvx_to_ripple
#endif

#define hvx_to_ripple(BS, N, T, x) hvx_to_ripple_v##N##T((BS), (x))
#define hvx_to_ripple_2d(BS, N, T, x) hvx_to_ripple_2d_v##N##T((BS), (x))

/// @brief Ripple to native HVX vector representation.
/// The Ripple block size needs to match
#ifdef __cplusplus
#define __decl_ripple_to_hvx(N_EL, C_T, T, PREC)                               \
  [[gnu::always_inline]] static v##N_EL##T##PREC                               \
      ripple_to_hvx_v##N_EL##T##PREC(ripple_block_t BS, const C_T &x) {        \
    C_T tmp[N_EL];                                                             \
    tmp[ripple_id(BS, 0)] = x;                                                 \
    return *((v##N_EL##T##PREC *)tmp);                                         \
  }                                                                            \
  [[gnu::always_inline]] static v##N_EL##T##PREC                               \
      ripple_to_hvx_2d_v##N_EL##T##PREC(ripple_block_t BS, const C_T &x) {     \
    C_T tmp[N_EL];                                                             \
    tmp[ripple_id(BS, 1) * ripple_get_block_size(BS, 0) + ripple_id(BS, 0)] =  \
        x;                                                                     \
    return *((v##N_EL##T##PREC *)tmp);                                         \
  }
#else // !__cplusplus
#define __decl_ripple_to_hvx(N_EL, C_T, T, PREC)                               \
  static __attribute__((always_inline)) v##N_EL##T##PREC                       \
      ripple_to_hvx_v##N_EL##T##PREC(ripple_block_t BS, const C_T x) {         \
    C_T tmp[N_EL];                                                             \
    tmp[ripple_id(BS, 0)] = x;                                                 \
    return *((v##N_EL##T##PREC *)tmp);                                         \
  }                                                                            \
  static __attribute__((always_inline)) v##N_EL##T##PREC                       \
      ripple_to_hvx_2d_v##N_EL##T##PREC(ripple_block_t BS, const C_T x) {      \
    C_T tmp[N_EL];                                                             \
    tmp[ripple_id(BS, 1) * ripple_get_block_size(BS, 0) + ripple_id(BS, 0)] =  \
        x;                                                                     \
    return *((v##N_EL##T##PREC *)tmp);                                         \
  }
#endif // __cplusplus

#define ripple_to_hvx(BS, N, T, x) ripple_to_hvx_v##N##T((BS), (x))
#define ripple_to_hvx_2d(BS, N, T, x) ripple_to_hvx_2d_v##N##T((BS), (x))

// Declare conversions for 1d and 2-d single- and double vectors. Vectors bigger
// than double cannot be returned in a register, instead, they use temporary
// memory location. Do not define functions for them yet.

__decl_ripple_to_hvx(128, int8_t, i, 8);
__decl_ripple_to_hvx(128, uint8_t, u, 8);
__decl_ripple_to_hvx(256, int8_t, i, 8);
__decl_ripple_to_hvx(256, uint8_t, u, 8);

__decl_ripple_to_hvx(64, int16_t, i, 16);
__decl_ripple_to_hvx(64, uint16_t, u, 16);
__decl_ripple_to_hvx(128, int16_t, i, 16);
__decl_ripple_to_hvx(128, uint16_t, u, 16);

__decl_ripple_to_hvx(32, int32_t, i, 32);
__decl_ripple_to_hvx(32, uint32_t, u, 32);
__decl_ripple_to_hvx(64, int32_t, i, 32);
__decl_ripple_to_hvx(64, uint32_t, u, 32);

__decl_ripple_to_hvx(32, float, f, 32);
__decl_ripple_to_hvx(64, float, f, 32);

__decl_ripple_to_hvx(16, int64_t, i, 64);
__decl_ripple_to_hvx(16, uint64_t, u, 64);
__decl_ripple_to_hvx(32, int64_t, i, 64);
__decl_ripple_to_hvx(32, uint64_t, u, 64);

__decl_ripple_to_hvx(16, double, f, 64);
__decl_ripple_to_hvx(32, double, f, 64);
#if __has_Float16__
__decl_ripple_to_hvx(64, _Float16, f, 16);
__decl_ripple_to_hvx(128, _Float16, f, 16);
#endif // __has_Float16__
#if __has_bf16__
__decl_ripple_to_hvx(64, __bf16, bf, 16);
__decl_ripple_to_hvx(128, __bf16, bf, 16);
#endif // __has_bf16__
#undef __decl_ripple_to_hvx

#endif // defined __hexagon__
