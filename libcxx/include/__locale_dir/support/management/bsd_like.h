//===-----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___LOCALE_DIR_SUPPORT_MANAGEMENT_BSD_LIKE_H
#define _LIBCPP___LOCALE_DIR_SUPPORT_MANAGEMENT_BSD_LIKE_H

#include <__config>
#include <clocale> // std::lconv

#include <xlocale.h> // uselocale() and others

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __locale {

using __locale_t = ::locale_t;

inline _LIBCPP_HIDE_FROM_ABI __locale_t __uselocale(__locale_t __loc) { return ::uselocale(__loc); }

inline _LIBCPP_HIDE_FROM_ABI __locale_t __newlocale(int __category_mask, const char* __locale, __locale_t __base) {
  return ::newlocale(__category_mask, __locale, __base);
}

inline _LIBCPP_HIDE_FROM_ABI void __freelocale(__locale_t __loc) { ::freelocale(__loc); }

inline _LIBCPP_HIDE_FROM_ABI lconv* __localeconv(__locale_t& __loc) { return ::localeconv_l(__loc); }

} // namespace __locale
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___LOCALE_DIR_SUPPORT_MANAGEMENT_BSD_LIKE_H
