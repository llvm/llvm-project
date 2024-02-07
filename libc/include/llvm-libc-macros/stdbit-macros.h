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
#else
#define stdc_leading_zeros(x)                                                  \
  _Generic((x),                                                                \
      unsigned char: stdc_leading_zeros_uc,                                    \
      unsigned short: stdc_leading_zeros_us,                                   \
      unsigned: stdc_leading_zeros_ui,                                         \
      unsigned long: stdc_leading_zeros_ul,                                    \
      unsigned long long: stdc_leading_zeros_ull)(x)
#define stdc_leading_ones(x)                                                   \
  _Generic((x),                                                                \
      unsigned char: stdc_leading_ones_uc,                                     \
      unsigned short: stdc_leading_ones_us,                                    \
      unsigned: stdc_leading_ones_ui,                                          \
      unsigned long: stdc_leading_ones_ul,                                     \
      unsigned long long: stdc_leading_ones_ull)(x)
#define stdc_trailing_zeros(x)                                                 \
  _Generic((x),                                                                \
      unsigned char: stdc_trailing_zeros_uc,                                   \
      unsigned short: stdc_trailing_zeros_us,                                  \
      unsigned: stdc_trailing_zeros_ui,                                        \
      unsigned long: stdc_trailing_zeros_ul,                                   \
      unsigned long long: stdc_trailing_zeros_ull)(x)
#define stdc_trailing_ones(x)                                                  \
  _Generic((x),                                                                \
      unsigned char: stdc_trailing_ones_uc,                                    \
      unsigned short: stdc_trailing_ones_us,                                   \
      unsigned: stdc_trailing_ones_ui,                                         \
      unsigned long: stdc_trailing_ones_ul,                                    \
      unsigned long long: stdc_trailing_ones_ull)(x)
#endif // __cplusplus

#endif // __LLVM_LIBC_MACROS_STDBIT_MACROS_H
