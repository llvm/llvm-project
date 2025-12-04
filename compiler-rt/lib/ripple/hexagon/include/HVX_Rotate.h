//==============================================================================
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//==============================================================================
//
// Part of the Ripple vector library to support the HVX rotate instruction.
//
//==============================================================================

#pragma once
#include "lib_func_attrib.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#else
extern {
#endif

#define _decl_hvx_rotate(ct, t) ct hvx_rotate_to_lower_##t(ct x, int32_t r)

_decl_hvx_rotate(int8_t, i8);
_decl_hvx_rotate(uint8_t, u8);
_decl_hvx_rotate(int16_t, i16);
_decl_hvx_rotate(uint16_t, u16);
#if __has_bf16__
_decl_hvx_rotate(__bf16, bf16);
#endif
#if __has_Float16__
_decl_hvx_rotate(_Float16, f16);
#endif
_decl_hvx_rotate(int32_t, i32);
_decl_hvx_rotate(uint32_t, u32);
_decl_hvx_rotate(float, f32);
_decl_hvx_rotate(int64_t, i64);
_decl_hvx_rotate(uint64_t, u64);
_decl_hvx_rotate(double, f64);

#undef _decl_hvx_rotate

} // extern "C" / extern

#ifndef __cplusplus

#if __has_bf16__
#if __has_Float16__
#define hvx_rotate_to_lower(x, n)                                              \
  _Generic((x), int8_t                                                         \
           : hvx_rotate_to_lower_i8((x), (n)), uint8_t                         \
           : hvx_rotate_to_lower_u8((x), (n)), int16_t                         \
           : hvx_rotate_to_lower_i16((x), (n)), uint16_t                       \
           : hvx_rotate_to_lower_u16((x), (n)), __bf16                         \
           : hvx_rotate_to_lower_bf16((x), (n)), _Float16                      \
           : hvx_rotate_to_lower_f16((x), (n)), int32_t                        \
           : hvx_rotate_to_lower_i32((x), (n)), uint32_t                       \
           : hvx_rotate_to_lower_u32((x), (n)), float                          \
           : hvx_rotate_to_lower_f32((x), (n)), int64_t                        \
           : hvx_rotate_to_lower_i64((x), (n)), uint64_t                       \
           : hvx_rotate_to_lower_u64((x), (n)), double                         \
           : hvx_rotate_to_lower_f64((x), (n)))

#else // !_Float16
#define hvx_rotate_to_lower(x, n)                                              \
  _Generic((x), int8_t                                                         \
           : hvx_rotate_to_lower_i8((x), (n)), uint8_t                         \
           : hvx_rotate_to_lower_u8((x), (n)), int16_t                         \
           : hvx_rotate_to_lower_i16((x), (n)), uint16_t                       \
           : hvx_rotate_to_lower_u16((x), (n)), __bf16                         \
           : hvx_rotate_to_lower_bf16((x), (n)), int32_t                       \
           : hvx_rotate_to_lower_i32((x), (n)), uint32_t                       \
           : hvx_rotate_to_lower_u32((x), (n)), float                          \
           : hvx_rotate_to_lower_f32((x), (n)), int64_t                        \
           : hvx_rotate_to_lower_i64((x), (n)), uint64_t                       \
           : hvx_rotate_to_lower_u64((x), (n)), double                         \
           : hvx_rotate_to_lower_f64((x), (n)))
#endif // __has_Float16__
#else  // !__has_bf16__
#if __has_Float16__
#define hvx_rotate_to_lower(x, n)                                              \
  _Generic((x), int8_t                                                         \
           : hvx_rotate_to_lower_i8((x), (n)), uint8_t                         \
           : hvx_rotate_to_lower_u8((x), (n)), int16_t                         \
           : hvx_rotate_to_lower_i16((x), (n)), uint16_t                       \
           : hvx_rotate_to_lower_u16((x), (n)), _Float16                       \
           : hvx_rotate_to_lower_f16((x), (n)), int32_t                        \
           : hvx_rotate_to_lower_i32((x), (n)), uint32_t                       \
           : hvx_rotate_to_lower_u32((x), (n)), float                          \
           : hvx_rotate_to_lower_f32((x), (n)), int64_t                        \
           : hvx_rotate_to_lower_i64((x), (n)), uint64_t                       \
           : hvx_rotate_to_lower_u64((x), (n)), double                         \
           : hvx_rotate_to_lower_f64((x), (n)))
#else // !__has_Float16__
#define hvx_rotate_to_lower(x, n)                                              \
  _Generic((x), int8_t                                                         \
           : hvx_rotate_to_lower_i8((x), (n)), uint8_t                         \
           : hvx_rotate_to_lower_u8((x), (n)), int16_t                         \
           : hvx_rotate_to_lower_i16((x), (n)), uint16_t                       \
           : hvx_rotate_to_lower_u16((x), (n)), int32_t                        \
           : hvx_rotate_to_lower_i32((x), (n)), uint32_t                       \
           : hvx_rotate_to_lower_u32((x), (n)), float                          \
           : hvx_rotate_to_lower_f32((x), (n)), int64_t                        \
           : hvx_rotate_to_lower_i64((x), (n)), uint64_t                       \
           : hvx_rotate_to_lower_u64((x), (n)), double                         \
           : hvx_rotate_to_lower_f64((x), (n)))
#endif // __has_Float16__
#endif // __has_bf16__

#else // !__cplusplus

#define _decl_hvx_rotate_spec(ct, t)                                           \
  [[gnu::always_inline]] static ct hvx_rotate_to_lower(ct x, int32_t n) {      \
    return hvx_rotate_to_lower_##t(x, n);                                      \
  }

_decl_hvx_rotate_spec(int8_t, i8);
_decl_hvx_rotate_spec(uint8_t, u8);
_decl_hvx_rotate_spec(int16_t, i16);
_decl_hvx_rotate_spec(uint16_t, u16);
_decl_hvx_rotate_spec(int32_t, i32);
_decl_hvx_rotate_spec(uint32_t, u32);
_decl_hvx_rotate_spec(int64_t, i64);
_decl_hvx_rotate_spec(uint64_t, u64);

_decl_hvx_rotate_spec(float, f32);
_decl_hvx_rotate_spec(double, f64);
#if __has_bf16__
_decl_hvx_rotate_spec(__bf16, bf16);
#endif
#if __has_Float16__
_decl_hvx_rotate_spec(_Float16, f16);
#endif

#undef _decl_hvx_rotate_spec

#endif