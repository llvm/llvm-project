//==============================================================================
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//==============================================================================
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//==============================================================================
//

#pragma once
#include "__ripple_vec.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define _decl_hvx_vdeal(ct, t)                                                 \
  __attribute__((used, always_inline, weak)) extern ct hvx_vdeal_##t(          \
      ct src, size_t chunk_size)

_decl_hvx_vdeal(int8_t, i8);
_decl_hvx_vdeal(int16_t, i16);
_decl_hvx_vdeal(int32_t, i32);
_decl_hvx_vdeal(int64_t, i64);

_decl_hvx_vdeal(uint8_t, u8);
_decl_hvx_vdeal(uint16_t, u16);
_decl_hvx_vdeal(uint32_t, u32);
_decl_hvx_vdeal(uint64_t, u64);

_decl_hvx_vdeal(double, f64);
_decl_hvx_vdeal(float, f32);
#if __has_bf16__
_decl_hvx_vdeal(__bf16, bf16);
#endif
#if __has_Float16__
_decl_hvx_vdeal(_Float16, f16);
#endif

#undef _decl_hvx_vdeal

#define _decl_hvx_vshuff(ct, t)                                                \
  __attribute__((used, always_inline, weak)) extern ct hvx_vshuff_##t(         \
      ct src, size_t chunk_size)

_decl_hvx_vshuff(int8_t, i8);
_decl_hvx_vshuff(int16_t, i16);
_decl_hvx_vshuff(int32_t, i32);
_decl_hvx_vshuff(int64_t, i64);

_decl_hvx_vshuff(uint8_t, u8);
_decl_hvx_vshuff(uint16_t, u16);
_decl_hvx_vshuff(uint32_t, u32);
_decl_hvx_vshuff(uint64_t, u64);

_decl_hvx_vshuff(double, f64);
_decl_hvx_vshuff(float, f32);
#if __has_bf16__
_decl_hvx_vshuff(__bf16, bf16);
#endif
#if __has_Float16__
_decl_hvx_vshuff(_Float16, f16);
#endif

#undef _decl_hvx_vshuff

#ifdef __cplusplus
} // extern "C"
#endif

#ifndef __cplusplus

#define __hvx_vdeal_anyint(X, Type, T, chunk_size, val)                        \
  ((Type)(sizeof(Type) == 1                                                    \
              ? hvx_vdeal_##T##8((val), (chunk_size))                          \
              : (sizeof(Type) == 2                                             \
                     ? hvx_vdeal_##T##16((val), (chunk_size))                  \
                     : (sizeof(Type) == 4                                      \
                            ? hvx_vdeal_##T##32((val), (chunk_size))           \
                            : hvx_vdeal_##T##64((val), (chunk_size))))))

#define __hvx_vshuff_anyint(Type, T, chunk_size, val)                          \
  ((Type)(sizeof(Type) == 1                                                    \
              ? hvx_vshuff_##T##8((val), (chunk_size))                         \
              : (sizeof(Type) == 2                                             \
                     ? hvx_vshuff_##T##16((val), (chunk_size))                 \
                     : (sizeof(Type) == 4                                      \
                            ? hvx_vshuff_##T##32((val), (chunk_size))          \
                            : hvx_vshuff_##T##64((val), (chunk_size))))))

#if __has_bf16__
#if __has_Float16__
#define hvx_vdeal(src, chunk_size)                                             \
  _Generic((src),                                                              \
      char: __ripple_char_is_signed                                            \
          ? __hvx_vdeal_anyint(char, i, chunk_size, src)                       \
          : __hvx_vdeal_anyint(char, u, chunk_size, src),                      \
      signed char: __hvx_vdeal_anyint(signed char, i, chunk_size, src),        \
      unsigned char: __hvx_vdeal_anyint(unsigned char, u, chunk_size, src),    \
      signed short: __hvx_vdeal_anyint(signed short, i, chunk_size, src),      \
      unsigned short: __hvx_vdeal_anyint(unsigned short, u, chunk_size, src),  \
      signed int: __hvx_vdeal_anyint(signed int, i, chunk_size, src),          \
      unsigned int: __hvx_vdeal_anyint(unsigned int, u, chunk_size, src),      \
      signed long: __hvx_vdeal_anyint(signed long, i, chunk_size, src),        \
      unsigned long: __hvx_vdeal_anyint(unsigned long, u, chunk_size, src),    \
      signed long long: __hvx_vdeal_anyint(signed long long, i, chunk_size,    \
                                           src),                               \
      unsigned long long: __hvx_vdeal_anyint(unsigned long long, u,            \
                                             chunk_size, src),                 \
      float: hvx_vdeal_f32((src), (chunk_size)),                               \
      double: hvx_vdeal_f64((src), (chunk_size)),                              \
      __bf16: hvx_vdeal_bf16((src), (chunk_size)),                             \
      _Float16: hvx_vdeal_f16((src), (chunk_size)))

#define hvx_vshuff(src, chunk_size)                                            \
  _Generic((src),                                                              \
      char: __ripple_char_is_signed                                            \
          ? __hvx_vshuff_anyint(char, i, chunk_size, src)                      \
          : __hvx_vshuff_anyint(char, u, chunk_size, src),                     \
      signed char: __hvx_vshuff_anyint(signed char, i, chunk_size, src),       \
      unsigned char: __hvx_vshuff_anyint(unsigned char, u, chunk_size, src),   \
      signed short: __hvx_vshuff_anyint(signed short, i, chunk_size, src),     \
      unsigned short: __hvx_vshuff_anyint(unsigned short, u, chunk_size, src), \
      signed int: __hvx_vshuff_anyint(signed int, i, chunk_size, src),         \
      unsigned int: __hvx_vshuff_anyint(unsigned int, u, chunk_size, src),     \
      signed long: __hvx_vshuff_anyint(signed long, i, chunk_size, src),       \
      unsigned long: __hvx_vshuff_anyint(unsigned long, u, chunk_size, src),   \
      signed long long: __hvx_vshuff_anyint(signed long long, i, chunk_size,   \
                                            src),                              \
      unsigned long long: __hvx_vshuff_anyint(unsigned long long, u,           \
                                              chunk_size, src),                \
      float: hvx_vshuff_f32((src), (chunk_size)),                              \
      double: hvx_vshuff_f64((src), (chunk_size)),                             \
      __bf16: hvx_vshuff_bf16((src), (chunk_size)),                            \
      _Float16: hvx_vshuff_f16((src), (chunk_size)))

#else // !_Float16
#define hvx_vdeal(src, chunk_size)                                             \
  _Generic((src),                                                              \
      char: __ripple_char_is_signed                                            \
          ? __hvx_vdeal_anyint(char, i, chunk_size, src)                       \
          : __hvx_vdeal_anyint(char, u, chunk_size, src),                      \
      signed char: __hvx_vdeal_anyint(signed char, i, chunk_size, src),        \
      unsigned char: __hvx_vdeal_anyint(unsigned char, u, chunk_size, src),    \
      signed short: __hvx_vdeal_anyint(signed short, i, chunk_size, src),      \
      unsigned short: __hvx_vdeal_anyint(unsigned short, u, chunk_size, src),  \
      signed int: __hvx_vdeal_anyint(signed int, i, chunk_size, src),          \
      unsigned int: __hvx_vdeal_anyint(unsigned int, u, chunk_size, src),      \
      signed long: __hvx_vdeal_anyint(signed long, i, chunk_size, src),        \
      unsigned long: __hvx_vdeal_anyint(unsigned long, u, chunk_size, src),    \
      signed long long: __hvx_vdeal_anyint(signed long long, i, chunk_size,    \
                                           src),                               \
      unsigned long long: __hvx_vdeal_anyint(                                  \
               unsigned long long, u, chunk_size,                              \
               src) float: hvx_vdeal_f32((src), (chunk_size)),                 \
      float: hvx_vdeal_f32((src), (chunk_size)),                               \
      double: hvx_vdeal_f64((src), (chunk_size)),                              \
      __bf16: hvx_vdeal_bf16((src), (chunk_size)))

#define hvx_vshuff(src, chunk_size)                                            \
  _Generic((src),                                                              \
      char: __ripple_char_is_signed                                            \
          ? __hvx_vshuff_anyint(char, i, chunk_size, src)                      \
          : __hvx_vshuff_anyint(char, u, chunk_size, src),                     \
      signed char: __hvx_vshuff_anyint(signed char, i, chunk_size, src),       \
      unsigned char: __hvx_vshuff_anyint(unsigned char, u, chunk_size, src),   \
      signed short: __hvx_vshuff_anyint(signed short, i, chunk_size, src),     \
      unsigned short: __hvx_vshuff_anyint(unsigned short, u, chunk_size, src), \
      signed int: __hvx_vshuff_anyint(signed int, i, chunk_size, src),         \
      unsigned int: __hvx_vshuff_anyint(unsigned int, u, chunk_size, src),     \
      signed long: __hvx_vshuff_anyint(signed long, i, chunk_size, src),       \
      unsigned long: __hvx_vshuff_anyint(unsigned long, u, chunk_size, src),   \
      signed long long: __hvx_vshuff_anyint(signed long long, i, chunk_size,   \
                                            src),                              \
      unsigned long long: __hvx_vshuff_anyint(unsigned long long, u,           \
                                              chunk_size, src),                \
      float: hvx_vshuff_f32((src), (chunk_size)),                              \
      double: hvx_vshuff_f64((src), (chunk_size)),                             \
      __bf16: hvx_vshuff_bf16((src), (chunk_size)))

#endif // __has_Float16__
#else  // !__has_bf16__
#if __has_Float16__
#define hvx_vdeal(src, chunk_size)                                             \
  _Generic((src),                                                              \
      char: __ripple_char_is_signed                                            \
          ? __hvx_vdeal_anyint(char, i, chunk_size, src)                       \
          : __hvx_vdeal_anyint(char, u, chunk_size, src),                      \
      signed char: __hvx_vdeal_anyint(signed char, i, chunk_size, src),        \
      unsigned char: __hvx_vdeal_anyint(unsigned char, u, chunk_size, src),    \
      signed short: __hvx_vdeal_anyint(signed short, i, chunk_size, src),      \
      unsigned short: __hvx_vdeal_anyint(unsigned short, u, chunk_size, src),  \
      signed int: __hvx_vdeal_anyint(signed int, i, chunk_size, src),          \
      unsigned int: __hvx_vdeal_anyint(unsigned int, u, chunk_size, src),      \
      signed long: __hvx_vdeal_anyint(signed long, i, chunk_size, src),        \
      unsigned long: __hvx_vdeal_anyint(unsigned long, u, chunk_size, src),    \
      signed long long: __hvx_vdeal_anyint(signed long long, i, chunk_size,    \
                                           src),                               \
      unsigned long long: __hvx_vdeal_anyint(unsigned long long, u,            \
                                             chunk_size, src),                 \
      float: hvx_vdeal_f32((src), (chunk_size)),                               \
      double: hvx_vdeal_f64((src), (chunk_size)),                              \
      _Float16: hvx_vdeal_f16((src), (chunk_size)))

#define hvx_vshuff(src, chunk_size)                                            \
  _Generic((src),                                                              \
      char: __ripple_char_is_signed                                            \
          ? __hvx_vshuff_anyint(char, i, chunk_size, src)                      \
          : __hvx_vshuff_anyint(char, u, chunk_size, src),                     \
      signed char: __hvx_vshuff_anyint(signed char, i, chunk_size, src),       \
      unsigned char: __hvx_vshuff_anyint(unsigned char, u, chunk_size, src),   \
      signed short: __hvx_vshuff_anyint(signed short, i, chunk_size, src),     \
      unsigned short: __hvx_vshuff_anyint(unsigned short, u, chunk_size, src), \
      signed int: __hvx_vshuff_anyint(signed int, i, chunk_size, src),         \
      unsigned int: __hvx_vshuff_anyint(unsigned int, u, chunk_size, src),     \
      signed long: __hvx_vshuff_anyint(signed long, i, chunk_size, src),       \
      unsigned long: __hvx_vshuff_anyint(unsigned long, u, chunk_size, src),   \
      signed long long: __hvx_vshuff_anyint(signed long long, i, chunk_size,   \
                                            src),                              \
      unsigned long long: __hvx_vshuff_anyint(unsigned long long, u,           \
                                              chunk_size, src),                \
      float: hvx_vshuff_f32((src), (chunk_size)),                              \
      double: hvx_vshuff_f64((src), (chunk_size)),                             \
      _Float16: hvx_vshuff_f16((src), (chunk_size)))

#else
#define hvx_vdeal(src, chunk_size)                                             \
  _Generic((src),                                                              \
      char: __ripple_char_is_signed                                            \
          ? __hvx_vdeal_anyint(char, i, chunk_size, src)                       \
          : __hvx_vdeal_anyint(char, u, chunk_size, src),                      \
      signed char: __hvx_vdeal_anyint(signed char, i, chunk_size, src),        \
      unsigned char: __hvx_vdeal_anyint(unsigned char, u, chunk_size, src),    \
      signed short: __hvx_vdeal_anyint(signed short, i, chunk_size, src),      \
      unsigned short: __hvx_vdeal_anyint(unsigned short, u, chunk_size, src),  \
      signed int: __hvx_vdeal_anyint(signed int, i, chunk_size, src),          \
      unsigned int: __hvx_vdeal_anyint(unsigned int, u, chunk_size, src),      \
      signed long: __hvx_vdeal_anyint(signed long, i, chunk_size, src),        \
      unsigned long: __hvx_vdeal_anyint(unsigned long, u, chunk_size, src),    \
      signed long long: __hvx_vdeal_anyint(signed long long, i, chunk_size,    \
                                           src),                               \
      unsigned long long: __hvx_vdeal_anyint(unsigned long long, u,            \
                                             chunk_size, src),                 \
      float: hvx_vdeal_f32((src), (chunk_size)),                               \
      double: hvx_vdeal_f64((src), (chunk_size)))

#define hvx_vshuff(src, chunk_size)                                            \
  _Generic((src),                                                              \
      char: __ripple_char_is_signed                                            \
          ? __hvx_vshuff_anyint(char, i, chunk_size, src)                      \
          : __hvx_vshuff_anyint(char, u, chunk_size, src),                     \
      signed char: __hvx_vshuff_anyint(signed char, i, chunk_size, src),       \
      unsigned char: __hvx_vshuff_anyint(unsigned char, u, chunk_size, src),   \
      signed short: __hvx_vshuff_anyint(signed short, i, chunk_size, src),     \
      unsigned short: __hvx_vshuff_anyint(unsigned short, u, chunk_size, src), \
      signed int: __hvx_vshuff_anyint(signed int, i, chunk_size, src),         \
      unsigned int: __hvx_vshuff_anyint(unsigned int, u, chunk_size, src),     \
      signed long: __hvx_vshuff_anyint(signed long, i, chunk_size, src),       \
      unsigned long: __hvx_vshuff_anyint(unsigned long, u, chunk_size, src),   \
      signed long long: __hvx_vshuff_anyint(signed long long, i, chunk_size,   \
                                            src),                              \
      unsigned long long: __hvx_vshuff_anyint(unsigned long long, u,           \
                                              chunk_size, src),                \
      float: hvx_vshuff_f32((src), (chunk_size)),                              \
      double: hvx_vshuff_f64((src), (chunk_size)))

#endif // __has_Float16__
#endif // __has_bf16__

#else // __cplusplus

#define _decl_hvx_vdeal_spec_int(CT, UI)                                       \
  [[gnu::always_inline]] static CT hvx_vdeal(CT Val, size_t chunk_size) {      \
    switch (sizeof(Val)) {                                                     \
    case 1:                                                                    \
      return hvx_vdeal_##UI##8((Val), (chunk_size));                           \
    case 2:                                                                    \
      return hvx_vdeal_##UI##16((Val), (chunk_size));                          \
    case 4:                                                                    \
      return hvx_vdeal_##UI##32((Val), (chunk_size));                          \
    case 8:                                                                    \
      return hvx_vdeal_##UI##64((Val), (chunk_size));                          \
    }                                                                          \
  }

#define _decl_hvx_vshuff_spec_int(CT, UI)                                      \
  [[gnu::always_inline]] static CT hvx_vshuff(CT Val, size_t chunk_size) {     \
    switch (sizeof(Val)) {                                                     \
    case 1:                                                                    \
      return hvx_vshuff_##UI##8((Val), (chunk_size));                          \
    case 2:                                                                    \
      return hvx_vshuff_##UI##16((Val), (chunk_size));                         \
    case 4:                                                                    \
      return hvx_vshuff_##UI##32((Val), (chunk_size));                         \
    case 8:                                                                    \
      return hvx_vshuff_##UI##64((Val), (chunk_size));                         \
    }                                                                          \
  }

[[gnu::always_inline]] static char hvx_vdeal(const char Val,
                                             size_t chunk_size) {
  return __ripple_char_is_signed ? hvx_vdeal_i8((Val), (chunk_size))
    : hvx_vdeal_u8((Val), (chunk_size));
}

_decl_hvx_vdeal_spec_int(signed char, i);
_decl_hvx_vdeal_spec_int(unsigned char, u);
_decl_hvx_vdeal_spec_int(short, i);
_decl_hvx_vdeal_spec_int(unsigned short, u);
_decl_hvx_vdeal_spec_int(int, i);
_decl_hvx_vdeal_spec_int(unsigned int, u);
_decl_hvx_vdeal_spec_int(long, i);
_decl_hvx_vdeal_spec_int(unsigned long, u);
_decl_hvx_vdeal_spec_int(long long, i);
_decl_hvx_vdeal_spec_int(unsigned long long, u);

#if __has_Float16__
[[gnu::always_inline]] static _Float16 hvx_vdeal(_Float16 Val,
                                                 size_t chunk_size) {
  return hvx_vdeal_f16((Val), (chunk_size));
}
#endif
#if __has_bf16__
[[gnu::always_inline]] static __bf16 hvx_vdeal(__bf16 Val,
                                               size_t chunk_size) {
  return hvx_vdeal_bf16((Val), (chunk_size));
}
#endif
[[gnu::always_inline]] static float hvx_vdeal(float Val,
                                              size_t chunk_size) {
  return hvx_vdeal_f32((Val), (chunk_size));
}
[[gnu::always_inline]] static double hvx_vdeal(double Val,
                                              size_t chunk_size) {
  return hvx_vdeal_f64((Val), (chunk_size));
}

_decl_hvx_vshuff_spec_int(signed char, i);
_decl_hvx_vshuff_spec_int(unsigned char, u);
_decl_hvx_vshuff_spec_int(short, i);
_decl_hvx_vshuff_spec_int(unsigned short, u);
_decl_hvx_vshuff_spec_int(int, i);
_decl_hvx_vshuff_spec_int(unsigned int, u);
_decl_hvx_vshuff_spec_int(long, i);
_decl_hvx_vshuff_spec_int(unsigned long, u);
_decl_hvx_vshuff_spec_int(long long, i);
_decl_hvx_vshuff_spec_int(unsigned long long, u);

[[gnu::always_inline]] static char hvx_vshuff(const char Val,
                                              size_t chunk_size) {
  return __ripple_char_is_signed ? hvx_vshuff_i8((Val), (chunk_size))
    : hvx_vshuff_u8((Val), (chunk_size));
}

#if __has_Float16__
[[gnu::always_inline]] static _Float16 hvx_vshuff(_Float16 Val,
                                                  size_t chunk_size) {
  return hvx_vshuff_f16((Val), (chunk_size));
}
#endif
#if __has_bf16__
[[gnu::always_inline]] static __bf16 hvx_vshuff(__bf16 Val,
                                                size_t chunk_size) {
  return hvx_vshuff_bf16((Val), (chunk_size));
}
#endif
[[gnu::always_inline]] static float hvx_vshuff(float Val,
                                               size_t chunk_size) {
  return hvx_vshuff_f32((Val), (chunk_size));
}
[[gnu::always_inline]] static double hvx_vshuff(double Val,
                                               size_t chunk_size) {
  return hvx_vshuff_f64((Val), (chunk_size));
}
#endif
