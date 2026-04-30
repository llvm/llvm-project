//===-----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___LOCALE_DIR_SUPPORT_LLVM_LIBC_H
#define _LIBCPP___LOCALE_DIR_SUPPORT_LLVM_LIBC_H

#include <__config>
#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <ctype.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __locale {

using __locale_t _LIBCPP_NODEBUG = ::locale_t;

#define _LIBCPP_COLLATE_MASK LC_COLLATE_MASK
#define _LIBCPP_CTYPE_MASK LC_CTYPE_MASK
#define _LIBCPP_MONETARY_MASK LC_MONETARY_MASK
#define _LIBCPP_NUMERIC_MASK LC_NUMERIC_MASK
#define _LIBCPP_TIME_MASK LC_TIME_MASK
#define _LIBCPP_MESSAGES_MASK LC_MESSAGES_MASK
#define _LIBCPP_ALL_MASK LC_ALL_MASK
#define _LIBCPP_LC_ALL LC_ALL

#if defined(_LIBCPP_BUILDING_LIBRARY)
using __lconv_t _LIBCPP_NODEBUG = std::lconv;

inline _LIBCPP_HIDE_FROM_ABI __locale_t __newlocale(int __category_mask, const char* __locale, __locale_t __base) {
  return ::newlocale(__category_mask, __locale, __base);
}

inline _LIBCPP_HIDE_FROM_ABI void __freelocale(__locale_t __loc) { ::freelocale(__loc); }

inline _LIBCPP_HIDE_FROM_ABI char* __setlocale(int __category, char const* __locale) {
  return ::setlocale(__category, __locale);
}

inline _LIBCPP_HIDE_FROM_ABI __lconv_t* __localeconv(__locale_t&) { return std::localeconv(); }
#endif // _LIBCPP_BUILDING_LIBRARY

} // namespace __locale
_LIBCPP_END_NAMESPACE_STD

#include <__locale_dir/support/no_locale/characters.h>
#include <__locale_dir/support/no_locale/conversions.h>
#include <__locale_dir/support/no_locale/formatting.h>
#include <__locale_dir/support/no_locale/strtonum.h>

#endif // _LIBCPP___LOCALE_DIR_SUPPORT_LLVM_LIBC_H
