//==============================================================================
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//==============================================================================
//
// Part of the Ripple vector library to support the HVX gather and scatter
// instructions.
//
//==============================================================================

#pragma once
#include "ripple.h"
#include <stddef.h>
#include <stdint.h>

// ___________________________ HVX vector gather _______________________________

#ifdef __cplusplus
extern "C" {
#endif

#define _decl_hvx_gather(ct, t)                                                \
  __attribute__((used, always_inline, weak)) extern void hvx_gather_##t(       \
      ct *dst, const ct *src, int index, size_t region_size)

_decl_hvx_gather(int8_t, i8);
_decl_hvx_gather(uint8_t, u8);
_decl_hvx_gather(int16_t, i16);
_decl_hvx_gather(uint16_t, u16);
_decl_hvx_gather(int32_t, i32);
_decl_hvx_gather(uint32_t, u32);
_decl_hvx_gather(int64_t, i64);
_decl_hvx_gather(uint64_t, u64);

_decl_hvx_gather(float, f32);
_decl_hvx_gather(double, f64);
#if __has_bf16__
_decl_hvx_gather(__bf16, bf16);
#endif
#if __has_Float16__
_decl_hvx_gather(_Float16, f16);
#endif
#undef _decl_hvx_gather

#define _decl_hvx_gather_16(ct, t)                                             \
  __attribute__((used, always_inline, weak)) extern void hvx_gather_##t##_16(  \
      ct *dst, const ct *src, int16_t index, size_t region_size)

_decl_hvx_gather_16(int8_t, i8);
_decl_hvx_gather_16(uint8_t, u8);

_decl_hvx_gather_16(int16_t, i16);
_decl_hvx_gather_16(uint16_t, u16);

#if __has_bf16__
_decl_hvx_gather_16(__bf16, bf16);
#endif
#if __has_Float16__
_decl_hvx_gather_16(_Float16, f16);
#endif

#undef _decl_hvx_gather_16

#ifdef __cplusplus
} // extern "C"
#endif

#ifndef __cplusplus
#define both_gathers(gather_fn, dst, src, index, region_size)                  \
  _Generic((index),                                                            \
      int16_t: gather_fn##_16((dst), (src), (index), (region_size)),           \
      int32_t: gather_fn((dst), (src), (index), (region_size)))

#if __has_bf16__
#define __extra_bf16_gather(dst, src, index, region_size)                      \
  , __bf16 : hvx_gather_bf16((dst), (src), (index), (region_size))
#else
#define __extra_bf16_gather(dst, src, index, region_size)
#endif

#if __has_Float16__
#define __extra_f16_gather(dst, src, index, region_size)                       \
  , _Float16 : hvx_gather_f16((dst), (src), (index), (region_size))
#else
#define __extra_f16_gather(dst, src, index, region_size)
#endif

#define hvx_gather(dst, src, index, region_size)                               \
  _Generic((src),                                                              \
      char *: hvx_gather_i8((dst), (src), (index), (region_size)),             \
      int16_t *: both_gathers(hvx_gather_i16, (dst), (src), (index),           \
                              (region_size)),                                  \
      int32_t *: hvx_gather_i32((dst), (src), (index), (region_size)),         \
      int64_t *: hvx_gather_i64((dst), (src), (index), (region_size)),         \
      uint8_t *: hvx_gather_u8((dst), (src), (index), (region_size)),          \
      uint16_t *: both_gathers(hvx_gather_u16, (dst), (src), (index),          \
                               (region_size)),                                 \
      uint32_t *: hvx_gather_u32((dst), (src), (index), (region_size)),        \
      uint64_t *: hvx_gather_u64((dst), (src), (index), (region_size)),        \
      float *: hvx_gather_f32((dst), (src), (index), (region_size)),           \
      double *: hvx_gather_f64((dst), (src), (index), (region_size))           \
          __extra_bf16_gather(dst, src, index, region_size)                    \
              __extra_f16_gather(dst, src, index, region_size))

#else // __cplusplus

#define _decl_hvx_gather_spec(ct, t)                                           \
  [[gnu::always_inline, gnu::used]] static void hvx_gather(                    \
      ct *dst, const ct *src, int index, size_t region_size) {                 \
    hvx_gather_##t(dst, src, index, region_size);                              \
  }

_decl_hvx_gather_spec(int8_t, i8);
_decl_hvx_gather_spec(uint8_t, u8);
_decl_hvx_gather_spec(int16_t, i16);
_decl_hvx_gather_spec(uint16_t, u16);
_decl_hvx_gather_spec(int32_t, i32);
_decl_hvx_gather_spec(uint32_t, u32);
_decl_hvx_gather_spec(int64_t, i64);
_decl_hvx_gather_spec(uint64_t, u64);

_decl_hvx_gather_spec(float, f32);
_decl_hvx_gather_spec(double, f64);
#if __has_bf16__
_decl_hvx_gather_spec(__bf16, bf16);
#endif
#if __has_Float16__
_decl_hvx_gather_spec(_Float16, f16);
#endif

#undef _decl_hvx_gather_spec

#define _decl_hvx_gather_16_spec(ct, t)                                        \
  [[gnu::always_inline, gnu::used]] static void hvx_gather(                    \
      ct *dst, const ct *src, int16_t index, size_t region_size) {             \
    hvx_gather_##t##_16(dst, src, index, region_size);                         \
  }

_decl_hvx_gather_16_spec(int16_t, i16);
_decl_hvx_gather_16_spec(uint16_t, u16);

_decl_hvx_gather_16_spec(int8_t, i8);
_decl_hvx_gather_16_spec(uint8_t, u8);

#if __has_bf16__
_decl_hvx_gather_16_spec(__bf16, bf16);
#endif
#if __has_Float16__
_decl_hvx_gather_16_spec(_Float16, f16);
#endif

#undef _decl_hvx_gather_16_spec

#endif // __cplusplus

// __________________________ HVX vector scatter _______________________________

#ifdef __cplusplus
extern "C" {
#endif
#define _decl_hvx_scatter(w)                                                   \
  __attribute__((used, always_inline, weak)) extern void hvx_scatter_i##w(     \
      int##w##_t *dst, int index, int##w##_t src, size_t region_size);         \
  __attribute__((used, always_inline, weak)) extern void hvx_scatter_u##w(     \
      uint##w##_t *dst, int index, uint##w##_t src, size_t region_size)

#define _decl_hvx_scatter_16(w)                                                \
  __attribute__((used, always_inline,                                          \
                 weak)) extern void hvx_scatter_i##w##_16(int##w##_t *dst,     \
                                                          int16_t index,       \
                                                          int##w##_t src,      \
                                                          size_t region_size); \
  __attribute__((used, always_inline,                                          \
                 weak)) extern void hvx_scatter_u##w##_16(uint##w##_t *dst,    \
                                                          int16_t index,       \
                                                          uint##w##_t src,     \
                                                          size_t region_size)

#define _decl_f_hvx_scatter(w, ct)                                             \
  __attribute__((used, always_inline, weak)) extern void hvx_scatter_f##w(     \
      ct *dst, int index, ct src, size_t region_size)

#define _decl_f_hvx_scatter_16(w, ct)                                          \
  __attribute__((used, always_inline,                                          \
                 weak)) extern void hvx_scatter_f##w##_16(ct *dst,             \
                                                          int16_t index,       \
                                                          ct src,              \
                                                          size_t region_size);


#define _decl_f_hvx_scatter_bf16 \
__attribute__((used, always_inline, weak)) extern void hvx_scatter_bf16( \
__bf16 *dst, int index, __bf16 src, size_t region_size)

#define _decl_f_hvx_scatter_bf16_16 \
__attribute__((used, always_inline, weak)) extern void hvx_scatter_bf16_16( \
__bf16 *dst, int16_t index, __bf16 src, size_t region_size)



_decl_hvx_scatter(8);
_decl_hvx_scatter(16);
_decl_hvx_scatter(32);
_decl_hvx_scatter(64);
_decl_hvx_scatter_16(8);
_decl_hvx_scatter_16(16);
#if __has_bf16__
_decl_f_hvx_scatter_bf16;
_decl_f_hvx_scatter_bf16_16;
#endif
#if __has_Float16__
_decl_f_hvx_scatter(16, _Float16);
_decl_f_hvx_scatter_16(16, _Float16);
#endif
_decl_f_hvx_scatter(32, float);
_decl_f_hvx_scatter(64, double);

#undef _decl_hvx_scatter
#undef _decl_hvx_scatter_16
#undef _decl_f_hvx_scatter

#ifdef __cplusplus
} // extern "C" / extern
#endif

#ifndef __cplusplus
#define both_scatters(scatter_fn, dst, index, src, region_size)                \
  _Generic((index),                                                            \
      int16_t: scatter_fn##_16((dst), (index), (src), (region_size)),          \
      int32_t: scatter_fn((dst), (index), (src), (region_size)))

#if __has_bf16__
#define __extra_bf16_scatter(dst, index, src, region_size)                     \
  , __bf16 : hvx_scatter_bf16((dst), (index), (src), (region_size))
#else
#define __extra_bf16_scatter(dst, index, src, region_size)
#endif

#if __has_Float16__
#define __extra_f16_scatter(dst, index, src, region_size)                      \
  , _Float16 : hvx_scatter_f16((dst), (index), (src), (region_size))
#else
#define __extra_f16_scatter(dst, index, src, region_size)
#endif

#define hvx_scatter(dst, index, src, region_size)                              \
  _Generic((src),                                                              \
      int8_t: hvx_scatter_i8((dst), (index), (src), (region_size)),            \
      int16_t: both_scatters(hvx_scatter_i16, dst, index, src, region_size),   \
      int32_t: hvx_scatter_i32((dst), (index), (src), (region_size)),          \
      int64_t: hvx_scatter_i64((dst), (index), (src), (region_size)),          \
      uint8_t: hvx_scatter_u8((dst), (index), (src), (region_size)),           \
      uint16_t: both_scatters(hvx_scatter_u16, dst, index, src, region_size),  \
      uint32_t: hvx_scatter_u32((dst), (index), (src), (region_size)),         \
      uint64_t: hvx_scatter_u64((dst), (index), (src), (region_size)),         \
      float: hvx_scatter_f32((dst), (index), (src), (region_size)),            \
      double: hvx_scatter_f64((dst), (index), (src), (region_size))            \
          __extra_bf16_scatter(dst, index, src, region_size)                   \
              __extra_f16_scatter(dst, index, src, region_size))

#else // __cplusplus
#define _decl_hvx_scatter_spec(ct, t)                                          \
  [[gnu::always_inline, gnu::used]] static void hvx_scatter(                   \
      ct *dst, int index, ct src, size_t region_size) {                        \
    hvx_scatter_##t(dst, index, src, region_size);                             \
  }

_decl_hvx_scatter_spec(int8_t, i8);
_decl_hvx_scatter_spec(uint8_t, u8);
_decl_hvx_scatter_spec(int16_t, i16);
_decl_hvx_scatter_spec(uint16_t, u16);
_decl_hvx_scatter_spec(int32_t, i32);
_decl_hvx_scatter_spec(uint32_t, u32);
_decl_hvx_scatter_spec(int64_t, i64);
_decl_hvx_scatter_spec(uint64_t, u64);

_decl_hvx_scatter_spec(float, f32);
_decl_hvx_scatter_spec(double, f64);
#if __has_bf16__
_decl_hvx_scatter_spec(__bf16, bf16);
#endif
#if __has_Float16__
_decl_hvx_scatter_spec(_Float16, f16);
#endif

#undef _decl_hvx_scatter_spec

#define _decl_hvx_scatter_16_spec(ct, t)                                       \
  [[gnu::always_inline, gnu::used]] static void hvx_scatter(                   \
      ct *dst, int16_t index, ct src, size_t region_size) {                    \
    hvx_scatter_##t##_16(dst, index, src, region_size);                        \
  }

_decl_hvx_scatter_16_spec(int8_t, i8);
_decl_hvx_scatter_16_spec(uint8_t, u8);

_decl_hvx_scatter_16_spec(int16_t, i16);
_decl_hvx_scatter_16_spec(uint16_t, u16);

#if __has_bf16__
_decl_hvx_scatter_16_spec(__bf16, bf16);
#endif
#if __has_Float16__
_decl_hvx_scatter_16_spec(_Float16, f16);
#endif

#undef _decl_hvx_scatter_16_spec

#endif // __cplusplus
