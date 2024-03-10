//===-- Definition of macros to be used with stdbit functions ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __LLVM_LIBC_MACROS_STDBIT_MACROS_H
#define __LLVM_LIBC_MACROS_STDBIT_MACROS_H

// TODO(https://github.com/llvm/llvm-project/issues/80509): support _BitInt().
#ifdef __cplusplus
inline unsigned stdc_leading_zeros(unsigned char x) {
  return stdc_leading_zeros_uc(x);
}
inline unsigned stdc_leading_zeros(unsigned short x) {
  return stdc_leading_zeros_us(x);
}
inline unsigned stdc_leading_zeros(unsigned x) {
  return stdc_leading_zeros_ui(x);
}
inline unsigned stdc_leading_zeros(unsigned long x) {
  return stdc_leading_zeros_ul(x);
}
inline unsigned stdc_leading_zeros(unsigned long long x) {
  return stdc_leading_zeros_ull(x);
}
inline unsigned stdc_leading_ones(unsigned char x) {
  return stdc_leading_ones_uc(x);
}
inline unsigned stdc_leading_ones(unsigned short x) {
  return stdc_leading_ones_us(x);
}
inline unsigned stdc_leading_ones(unsigned x) {
  return stdc_leading_ones_ui(x);
}
inline unsigned stdc_leading_ones(unsigned long x) {
  return stdc_leading_ones_ul(x);
}
inline unsigned stdc_leading_ones(unsigned long long x) {
  return stdc_leading_ones_ull(x);
}
inline unsigned stdc_trailing_zeros(unsigned char x) {
  return stdc_trailing_zeros_uc(x);
}
inline unsigned stdc_trailing_zeros(unsigned short x) {
  return stdc_trailing_zeros_us(x);
}
inline unsigned stdc_trailing_zeros(unsigned x) {
  return stdc_trailing_zeros_ui(x);
}
inline unsigned stdc_trailing_zeros(unsigned long x) {
  return stdc_trailing_zeros_ul(x);
}
inline unsigned stdc_trailing_zeros(unsigned long long x) {
  return stdc_trailing_zeros_ull(x);
}
inline unsigned stdc_trailing_ones(unsigned char x) {
  return stdc_trailing_ones_uc(x);
}
inline unsigned stdc_trailing_ones(unsigned short x) {
  return stdc_trailing_ones_us(x);
}
inline unsigned stdc_trailing_ones(unsigned x) {
  return stdc_trailing_ones_ui(x);
}
inline unsigned stdc_trailing_ones(unsigned long x) {
  return stdc_trailing_ones_ul(x);
}
inline unsigned stdc_trailing_ones(unsigned long long x) {
  return stdc_trailing_ones_ull(x);
}
inline unsigned stdc_first_leading_zero(unsigned char x) {
  return stdc_first_leading_zero_uc(x);
}
inline unsigned stdc_first_leading_zero(unsigned short x) {
  return stdc_first_leading_zero_us(x);
}
inline unsigned stdc_first_leading_zero(unsigned x) {
  return stdc_first_leading_zero_ui(x);
}
inline unsigned stdc_first_leading_zero(unsigned long x) {
  return stdc_first_leading_zero_ul(x);
}
inline unsigned stdc_first_leading_zero(unsigned long long x) {
  return stdc_first_leading_zero_ull(x);
}
inline unsigned stdc_first_leading_one(unsigned char x) {
  return stdc_first_leading_one_uc(x);
}
inline unsigned stdc_first_leading_one(unsigned short x) {
  return stdc_first_leading_one_us(x);
}
inline unsigned stdc_first_leading_one(unsigned x) {
  return stdc_first_leading_one_ui(x);
}
inline unsigned stdc_first_leading_one(unsigned long x) {
  return stdc_first_leading_one_ul(x);
}
inline unsigned stdc_first_leading_one(unsigned long long x) {
  return stdc_first_leading_one_ull(x);
}
inline unsigned stdc_first_trailing_zero(unsigned char x) {
  return stdc_first_trailing_zero_uc(x);
}
inline unsigned stdc_first_trailing_zero(unsigned short x) {
  return stdc_first_trailing_zero_us(x);
}
inline unsigned stdc_first_trailing_zero(unsigned x) {
  return stdc_first_trailing_zero_ui(x);
}
inline unsigned stdc_first_trailing_zero(unsigned long x) {
  return stdc_first_trailing_zero_ul(x);
}
inline unsigned stdc_first_trailing_zero(unsigned long long x) {
  return stdc_first_trailing_zero_ull(x);
}
inline unsigned stdc_first_trailing_one(unsigned char x) {
  return stdc_first_trailing_one_uc(x);
}
inline unsigned stdc_first_trailing_one(unsigned short x) {
  return stdc_first_trailing_one_us(x);
}
inline unsigned stdc_first_trailing_one(unsigned x) {
  return stdc_first_trailing_one_ui(x);
}
inline unsigned stdc_first_trailing_one(unsigned long x) {
  return stdc_first_trailing_one_ul(x);
}
inline unsigned stdc_first_trailing_one(unsigned long long x) {
  return stdc_first_trailing_one_ull(x);
}
inline unsigned stdc_count_zeros(unsigned char x) {
  return stdc_count_zeros_uc(x);
}
inline unsigned stdc_count_zeros(unsigned short x) {
  return stdc_count_zeros_us(x);
}
inline unsigned stdc_count_zeros(unsigned x) { return stdc_count_zeros_ui(x); }
inline unsigned stdc_count_zeros(unsigned long x) {
  return stdc_count_zeros_ul(x);
}
inline unsigned stdc_count_zeros(unsigned long long x) {
  return stdc_count_zeros_ull(x);
}
inline unsigned stdc_count_ones(unsigned char x) {
  return stdc_count_ones_uc(x);
}
inline unsigned stdc_count_ones(unsigned short x) {
  return stdc_count_ones_us(x);
}
inline unsigned stdc_count_ones(unsigned x) { return stdc_count_ones_ui(x); }
inline unsigned stdc_count_ones(unsigned long x) {
  return stdc_count_ones_ul(x);
}
inline unsigned stdc_count_ones(unsigned long long x) {
  return stdc_count_ones_ull(x);
}
inline bool stdc_has_single_bit(unsigned char x) {
  return stdc_has_single_bit_uc(x);
}
inline bool stdc_has_single_bit(unsigned short x) {
  return stdc_has_single_bit_us(x);
}
inline bool stdc_has_single_bit(unsigned x) {
  return stdc_has_single_bit_ui(x);
}
inline bool stdc_has_single_bit(unsigned long x) {
  return stdc_has_single_bit_ul(x);
}
inline bool stdc_has_single_bit(unsigned long long x) {
  return stdc_has_single_bit_ull(x);
}
inline unsigned stdc_bit_width(unsigned char x) { return stdc_bit_width_uc(x); }
inline unsigned stdc_bit_width(unsigned short x) {
  return stdc_bit_width_us(x);
}
inline unsigned stdc_bit_width(unsigned x) { return stdc_bit_width_ui(x); }
inline unsigned stdc_bit_width(unsigned long x) { return stdc_bit_width_ul(x); }
inline unsigned stdc_bit_width(unsigned long long x) {
  return stdc_bit_width_ull(x);
}
inline unsigned char stdc_bit_floor(unsigned char x) {
  return stdc_bit_floor_uc(x);
}
inline unsigned short stdc_bit_floor(unsigned short x) {
  return stdc_bit_floor_us(x);
}
inline unsigned stdc_bit_floor(unsigned x) { return stdc_bit_floor_ui(x); }
inline unsigned long stdc_bit_floor(unsigned long x) {
  return stdc_bit_floor_ul(x);
}
inline unsigned long long stdc_bit_floor(unsigned long long x) {
  return stdc_bit_floor_ull(x);
}
inline unsigned char stdc_bit_ceil(unsigned char x) {
  return stdc_bit_ceil_uc(x);
}
inline unsigned short stdc_bit_ceil(unsigned short x) {
  return stdc_bit_ceil_us(x);
}
inline unsigned stdc_bit_ceil(unsigned x) { return stdc_bit_ceil_ui(x); }
inline unsigned long stdc_bit_ceil(unsigned long x) {
  return stdc_bit_ceil_ul(x);
}
inline unsigned long long stdc_bit_ceil(unsigned long long x) {
  return stdc_bit_ceil_ull(x);
}
#else

#define STDBIT_GENERIC_FUNC(FUNC_NAME, x)                                      \
  _Generic((x),                                                                \
      unsigned char: FUNC_NAME##_uc,                                           \
      unsigned short: FUNC_NAME##_us,                                          \
      unsigned: FUNC_NAME##_ui,                                                \
      unsigned long: FUNC_NAME##_ul,                                           \
      unsigned long long: FUNC_NAME##_ull)(x)

#define stdc_leading_zeros(x) STDBIT_GENERIC_FUNC(stdc_leading_zeros, (x))
#define stdc_leading_ones(x) STDBIT_GENERIC_FUNC(stdc_leading_ones, (x))
#define stdc_trailing_zeros(x) STDBIT_GENERIC_FUNC(stdc_trailing_zeros, (x))
#define stdc_trailing_ones(x) STDBIT_GENERIC_FUNC(stdc_trailing_ones, (x))
#define stdc_first_leading_zero(x)                                             \
  STDBIT_GENERIC_FUNC(stdc_first_leading_zero, (x))
#define stdc_first_leading_one(x)                                              \
  STDBIT_GENERIC_FUNC(stdc_first_leading_one, (x))
#define stdc_first_trailing_zero(x)                                            \
  STDBIT_GENERIC_FUNC(stdc_first_trailing_zero, (x))
#define stdc_first_trailing_one(x)                                             \
  STDBIT_GENERIC_FUNC(stdc_first_trailing_one, (x))
#define stdc_count_zeros(x) STDBIT_GENERIC_FUNC(stdc_count_zeros, (x))
#define stdc_count_ones(x) STDBIT_GENERIC_FUNC(stdc_count_ones, (x))
#define stdc_has_single_bit(x) STDBIT_GENERIC_FUNC(stdc_has_single_bit, (x))
#define stdc_bit_width(x) STDBIT_GENERIC_FUNC(stdc_bit_width, (x))
#define stdc_bit_floor(x) STDBIT_GENERIC_FUNC(stdc_bit_floor, (x))
#define stdc_bit_ceil(x) STDBIT_GENERIC_FUNC(stdc_bit_ceil, (x))

#undef STDBIT_GENERIC_FUNC

#endif // __cplusplus

#endif // __LLVM_LIBC_MACROS_STDBIT_MACROS_H
