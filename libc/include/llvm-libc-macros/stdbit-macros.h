//===-- Definition of macros to be used with stdbit functions ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __LLVM_LIBC_MACROS_STDBIT_MACROS_H
#define __LLVM_LIBC_MACROS_STDBIT_MACROS_H

#ifdef __cplusplus
inline unsigned char stdc_leading_zeros(unsigned char x) {
  return stdc_leading_zeros_uc(x);
}
inline unsigned short stdc_leading_zeros(unsigned short x) {
  return stdc_leading_zeros_us(x);
}
inline unsigned stdc_leading_zeros(unsigned x) {
  return stdc_leading_zeros_ui(x);
}
inline unsigned long stdc_leading_zeros(unsigned long x) {
  return stdc_leading_zeros_ul(x);
}
inline unsigned long long stdc_leading_zeros(unsigned long long x) {
  return stdc_leading_zeros_ull(x);
}
#else
#define stdc_leading_zeros(x)                                                  \
  _Generic((x),                                                                \
      unsigned char: stdc_leading_zeros_uc,                                    \
      unsigned short: stdc_leading_zeros_us,                                   \
      unsigned: stdc_leading_zeros_ui,                                         \
      unsigned long: stdc_leading_zeros_ul,                                    \
      unsigned long long: stdc_leading_zeros_ull)(x)
#endif // __cplusplus

#endif // __LLVM_LIBC_MACROS_STDBIT_MACROS_H
