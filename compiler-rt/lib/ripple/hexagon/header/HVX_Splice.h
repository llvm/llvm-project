//==============================================================================
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//==============================================================================

#pragma once
#include "__ripple_vec.h"
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

#define _decl_hvx_splice(ct, t)                                                \
    __attribute__((used, always_inline, weak)) extern ct hvx_splice_##t(       \
        ct left, ct right, size_t start);                                      \
    __attribute__((used, always_inline, weak)) extern ct hvx_lsplice_##t(      \
        ct left, ct right, size_t start);

_decl_hvx_splice(int8_t, i8);
_decl_hvx_splice(int16_t, i16);
_decl_hvx_splice(int32_t, i32);
_decl_hvx_splice(int64_t, i64);
_decl_hvx_splice(uint8_t, u8);
_decl_hvx_splice(uint16_t, u16);
_decl_hvx_splice(uint32_t, u32);
_decl_hvx_splice(uint64_t, u64);
_decl_hvx_splice(double, f64);
_decl_hvx_splice(float, f32);
#if __has_bf16__
_decl_hvx_splice(__bf16, bf16);
#endif
#if __has_Float16__
_decl_hvx_splice(_Float16, f16);
#endif
#undef _decl_hvx_splice
#ifdef __cplusplus
} // extern "C"
#endif
#ifndef __cplusplus
#if __has_bf16__
#define __extra_bf16_splice(left, right, start)                                \
  __bf16 : hvx_splice_bf16(left, right, start)
#define __extra_bf16_lsplice(left, right, start)                               \
  __bf16 : hvx_lsplice_bf16(left, right, start)
#else
#define __extra_bf16_splice(left, right, start)
#define __extra_bf16_lsplice(left, right, start)
#endif
#if __has_Float16__
#define __extra_f16_splice(left, right, start)                                 \
  _Float16 : hvx_splice_f16(left, right, start)
#define __extra_f16_lsplice(left, right, start)                                \
  _Float16 : hvx_lsplice_f16(left, right, start)
#else
#define __extra_f16_splice(left, right, start)
#define __extra_f16_lsplice(left, right, start)
#endif
#define __hvx_splice_anyint(Type, T, start, left, right)                       \
  ((Type)(sizeof(Type) == 1                                                    \
              ? hvx_splice_##T##8((left), (right), (start))                    \
              : (sizeof(Type) == 2                                             \
                     ? hvx_splice_##T##16((left), (right), (start))            \
                     : (sizeof(Type) == 4                                      \
                            ? hvx_splice_##T##32((left), (right), (start))     \
                            : hvx_splice_##T##64((left), (right), (start))))))
#define __hvx_lsplice_anyint(Type, T, start, left, right)                      \
  ((Type)(sizeof(Type) == 1                                                    \
              ? hvx_lsplice_##T##8((left), (right), (start))                   \
              : (sizeof(Type) == 2                                             \
                     ? hvx_lsplice_##T##16((left), (right), (start))           \
                     : (sizeof(Type) == 4                                      \
                            ? hvx_lsplice_##T##32((left), (right), (start))    \
                            : hvx_lsplice_##T##64((left), (right),             \
                                                  (start))))))
#define hvx_splice(left, right, start)                                          \
  _Generic((left),                                                             \
      char: __ripple_char_is_signed                                            \
          ? __hvx_splice_anyint(char, i, start, left, right)                    \
          : __hvx_splice_anyint(char, u, start, left, right),                   \
      signed char: __hvx_splice_anyint(signed char, i, start, left, right),     \
      unsigned char: __hvx_splice_anyint(unsigned char, u, start, left, right), \
      signed short: __hvx_splice_anyint(signed short, i, start, left, right),   \
      unsigned short: __hvx_splice_anyint(unsigned short, u, start, left,       \
                                         right),                               \
      signed int: __hvx_splice_anyint(signed int, i, start, left, right),       \
      unsigned int: __hvx_splice_anyint(unsigned int, u, start, left, right),   \
      signed long: __hvx_splice_anyint(signed long, i, start, left, right),     \
      unsigned long: __hvx_splice_anyint(unsigned long, u, start, left, right), \
      signed long long: __hvx_splice_anyint(signed long long, i, start, left,   \
                                           right),                             \
      unsigned long long: __hvx_splice_anyint(unsigned long long, u, start,     \
                                             left, right),                     \
      float: hvx_splice_f32(left, right, start),                                \
      double: hvx_splice_f64(left, right, start),                               \
      __extra_bf16_splice(left, right, start),                              \
      __extra_f16_splice(left, right, start)

#define hvx_lsplice(left, right, start)                                        \
  _Generic((left),                                                             \
      char: __ripple_char_is_signed                                            \
          ? __hvx_lsplice_anyint(char, i, start, left, right)                   \
          : __hvx_lsplice_anyint(char, u, start, left, right),                  \
      signed char: __hvx_lsplice_anyint(signed char, i, start, left, right),    \
      unsigned char: __hvx_lsplice_anyint(unsigned char, u, start, left,        \
                                         right),                               \
      signed short: __hvx_lsplice_anyint(signed short, i, start, left, right),  \
      unsigned short: __hvx_lsplice_anyint(unsigned short, u, start, left,      \
                                          right),                              \
      signed int: __hvx_lsplice_anyint(signed int, i, start, left, right),      \
      unsigned int: __hvx_lsplice_anyint(unsigned int, u, start, left, right),  \
      signed long: __hvx_lsplice_anyint(signed long, i, start, left, right),    \
      unsigned long: __hvx_lsplice_anyint(unsigned long, u, start, left,        \
                                         right),                               \
      signed long long: __hvx_lsplice_anyint(signed long long, i, start, left,  \
                                            right),                            \
      unsigned long long: __hvx_lsplice_anyint(unsigned long long, u, start,    \
                                              left, right),                    \
      float: hvx_lsplice_f32(left, right, start),                               \
      double: hvx_lsplice_f64(left, right, start),                              \
      __extra_bf16_lsplice(left, right, start),                              \
      __extra_f16_lsplice(left, right, start)
#else // __cplusplus
#define _decl_hvx_splice_spec_int(CT, UI)                                      \
  [[gnu::always_inline]] static CT hvx_splice(CT left, CT right,               \
                                              size_t start) {                  \
    switch (sizeof(left)) {                                                    \
    case 1:                                                                    \
      return hvx_splice_##UI##8((left), (right), (start));                     \
    case 2:                                                                    \
      return hvx_splice_##UI##16((left), (right), (start));                    \
    case 4:                                                                    \
      return hvx_splice_##UI##32((left), (right), (start));                    \
    case 8:                                                                    \
      return hvx_splice_##UI##64((left), (right), (start));                    \
    default:                                                                   \
      __builtin_unreachable();                                                 \
    }                                                                          \
  }
#define _decl_hvx_lsplice_spec_int(CT, UI)                                     \
  [[gnu::always_inline]] static CT hvx_lsplice(CT left, CT right,              \
                                               size_t start) {                 \
    switch (sizeof(left)) {                                                    \
    case 1:                                                                    \
      return hvx_lsplice_##UI##8((left), (right), (start));                    \
    case 2:                                                                    \
      return hvx_lsplice_##UI##16((left), (right), (start));                   \
    case 4:                                                                    \
      return hvx_lsplice_##UI##32((left), (right), (start));                   \
    case 8:                                                                    \
      return hvx_lsplice_##UI##64((left), (right), (start));                   \
    default:                                                                   \
      __builtin_unreachable();                                                 \
    }                                                                          \
  }
#if __has_Float16__
[[gnu::always_inline]] static _Float16 hvx_splice(_Float16 left, _Float16 right,
                                                  size_t start) {
  return hvx_splice_f16((left), (right), (start));
}
#endif
#if __has_bf16__
[[gnu::always_inline]] static __bf16 hvx_splice(__bf16 left, __bf16 right,
                                                size_t start) {
  return hvx_splice_bf16((left), (right), (start));
}
#endif
[[gnu::always_inline]] static float hvx_splice(float left, float right,
                                               size_t start) {
  return hvx_splice_f32((left), (right), (start));
}
[[gnu::always_inline]] static double hvx_splice(double left, double right,
                                                size_t start) {
  return hvx_splice_f64((left), (right), (start));
}
[[gnu::always_inline]] static char hvx_splice(const char left, const char right,
                                              size_t start) {
  return __ripple_char_is_signed ? hvx_splice_i8((left), (right), (start))
                                 : hvx_splice_u8((left), (right), (start));
}
_decl_hvx_splice_spec_int(signed char, i);
_decl_hvx_splice_spec_int(unsigned char, u);
_decl_hvx_splice_spec_int(short, i);
_decl_hvx_splice_spec_int(unsigned short, u);
_decl_hvx_splice_spec_int(int, i);
_decl_hvx_splice_spec_int(unsigned int, u);
_decl_hvx_splice_spec_int(long, i);
_decl_hvx_splice_spec_int(unsigned long, u);
_decl_hvx_splice_spec_int(long long, i);
_decl_hvx_splice_spec_int(unsigned long long, u);
#if __has_Float16__
[[gnu::always_inline]] static _Float16
hvx_lsplice(_Float16 left, _Float16 right, size_t start) {
  return hvx_lsplice_f16((left), (right), (start));
}
#endif
#if __has_bf16__
[[gnu::always_inline]] static __bf16 hvx_lsplice(__bf16 left, __bf16 right,
                                                 size_t start) {
  return hvx_lsplice_bf16((left), (right), (start));
}
#endif
[[gnu::always_inline]] static float hvx_lsplice(float left, float right,
                                                size_t start) {
  return hvx_lsplice_f32((left), (right), (start));
}
[[gnu::always_inline]] static double hvx_lsplice(double left, double right,
                                                 size_t start) {
  return hvx_lsplice_f64((left), (right), (start));
}
[[gnu::always_inline]] static char hvx_lsplice(const char left,
                                               const char right, size_t start) {
  return __ripple_char_is_signed ? hvx_lsplice_i8((left), (right), (start))
                                 : hvx_lsplice_u8((left), (right), (start));
}
_decl_hvx_lsplice_spec_int(signed char, i);
_decl_hvx_lsplice_spec_int(unsigned char, u);
_decl_hvx_lsplice_spec_int(short, i);
_decl_hvx_lsplice_spec_int(unsigned short, u);
_decl_hvx_lsplice_spec_int(int, i);
_decl_hvx_lsplice_spec_int(unsigned int, u);
_decl_hvx_lsplice_spec_int(long, i);
_decl_hvx_lsplice_spec_int(unsigned long, u);
_decl_hvx_lsplice_spec_int(long long, i);
_decl_hvx_lsplice_spec_int(unsigned long long, u);
#endif