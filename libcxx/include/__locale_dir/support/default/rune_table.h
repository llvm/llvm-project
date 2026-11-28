//===-----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___LOCALE_DIR_SUPPORT_DEFAULT_RUNE_TABLE_H
#define _LIBCPP___LOCALE_DIR_SUPPORT_DEFAULT_RUNE_TABLE_H

#include <__config>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

namespace __locale {
struct __ctype_base {
  typedef unsigned long mask;
  static const mask space  = 1 << 0;
  static const mask print  = 1 << 1;
  static const mask cntrl  = 1 << 2;
  static const mask upper  = 1 << 3;
  static const mask lower  = 1 << 4;
  static const mask alpha  = 1 << 5;
  static const mask digit  = 1 << 6;
  static const mask punct  = 1 << 7;
  static const mask xdigit = 1 << 8;
  static const mask blank  = 1 << 9;
#if defined(__BIONIC__)
  // Historically this was a part of regex_traits rather than ctype_base. The
  // historical value of the constant is preserved for ABI compatibility.
  static const mask __regex_word = 0x8000;
#else
  static const mask __regex_word = 1 << 10;
#endif // defined(__BIONIC__)
};

#ifdef _LIBCPP_BUILDING_LIBRARY
inline const __ctype_base::mask* __classic_table() noexcept {
  constexpr auto space  = __ctype_base::space;
  constexpr auto print  = __ctype_base::print;
  constexpr auto cntrl  = __ctype_base::cntrl;
  constexpr auto upper  = __ctype_base::upper;
  constexpr auto lower  = __ctype_base::lower;
  constexpr auto alpha  = __ctype_base::alpha;
  constexpr auto digit  = __ctype_base::digit;
  constexpr auto punct  = __ctype_base::punct;
  constexpr auto xdigit = __ctype_base::xdigit;
  constexpr auto blank  = __ctype_base::blank;

  // clang-format off
    static constexpr const __ctype_base::mask __table[256] = {
        cntrl,                          cntrl,
        cntrl,                          cntrl,
        cntrl,                          cntrl,
        cntrl,                          cntrl,
        cntrl,                          cntrl | space | blank,
        cntrl | space,                  cntrl | space,
        cntrl | space,                  cntrl | space,
        cntrl,                          cntrl,
        cntrl,                          cntrl,
        cntrl,                          cntrl,
        cntrl,                          cntrl,
        cntrl,                          cntrl,
        cntrl,                          cntrl,
        cntrl,                          cntrl,
        cntrl,                          cntrl,
        cntrl,                          cntrl,
        space | blank | print,          punct | print,
        punct | print,                  punct | print,
        punct | print,                  punct | print,
        punct | print,                  punct | print,
        punct | print,                  punct | print,
        punct | print,                  punct | print,
        punct | print,                  punct | print,
        punct | print,                  punct | print,
        digit | print | xdigit,         digit | print | xdigit,
        digit | print | xdigit,         digit | print | xdigit,
        digit | print | xdigit,         digit | print | xdigit,
        digit | print | xdigit,         digit | print | xdigit,
        digit | print | xdigit,         digit | print | xdigit,
        punct | print,                  punct | print,
        punct | print,                  punct | print,
        punct | print,                  punct | print,
        punct | print,                  upper | xdigit | print | alpha,
        upper | xdigit | print | alpha, upper | xdigit | print | alpha,
        upper | xdigit | print | alpha, upper | xdigit | print | alpha,
        upper | xdigit | print | alpha, upper | print | alpha,
        upper | print | alpha,          upper | print | alpha,
        upper | print | alpha,          upper | print | alpha,
        upper | print | alpha,          upper | print | alpha,
        upper | print | alpha,          upper | print | alpha,
        upper | print | alpha,          upper | print | alpha,
        upper | print | alpha,          upper | print | alpha,
        upper | print | alpha,          upper | print | alpha,
        upper | print | alpha,          upper | print | alpha,
        upper | print | alpha,          upper | print | alpha,
        upper | print | alpha,          punct | print,
        punct | print,                  punct | print,
        punct | print,                  punct | print,
        punct | print,                  lower | xdigit | print | alpha,
        lower | xdigit | print | alpha, lower | xdigit | print | alpha,
        lower | xdigit | print | alpha, lower | xdigit | print | alpha,
        lower | xdigit | print | alpha, lower | print | alpha,
        lower | print | alpha,          lower | print | alpha,
        lower | print | alpha,          lower | print | alpha,
        lower | print | alpha,          lower | print | alpha,
        lower | print | alpha,          lower | print | alpha,
        lower | print | alpha,          lower | print | alpha,
        lower | print | alpha,          lower | print | alpha,
        lower | print | alpha,          lower | print | alpha,
        lower | print | alpha,          lower | print | alpha,
        lower | print | alpha,          lower | print | alpha,
        lower | print | alpha,          punct | print,
        punct | print,                  punct | print,
        punct | print,                  cntrl,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };
  // clang-format on
  return __table;
}
#endif

} // namespace __locale

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___LOCALE_DIR_SUPPORT_DEFAULT_RUNE_TABLE_H
