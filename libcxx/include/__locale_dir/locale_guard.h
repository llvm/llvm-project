//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___LOCALE_DIR_LOCALE_GUARD_H
#define _LIBCPP___LOCALE_DIR_LOCALE_GUARD_H

#include <__config>
#include <__locale> // for locale_t
#include <clocale>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if defined(_LIBCPP_MSVCRT_LIKE)
struct __locale_guard {
  __locale_guard(__locale::__locale_t __l) : __status(_configthreadlocale(_ENABLE_PER_THREAD_LOCALE)) {
    // Setting the locale can be expensive even when the locale given is
    // already the current locale, so do an explicit check to see if the
    // current locale is already the one we want.
    const char* __lc = __setlocale(nullptr);
    // If every category is the same, the locale string will simply be the
    // locale name, otherwise it will be a semicolon-separated string listing
    // each category.  In the second case, we know at least one category won't
    // be what we want, so we only have to check the first case.
    if (std::strcmp(__l.__get_locale(), __lc) != 0) {
      __locale_all = _strdup(__lc);
      if (__locale_all == nullptr)
        __throw_bad_alloc();
      __setlocale(__l.__get_locale());
    }
  }
  ~__locale_guard() {
    // The CRT documentation doesn't explicitly say, but setlocale() does the
    // right thing when given a semicolon-separated list of locale settings
    // for the different categories in the same format as returned by
    // setlocale(LC_ALL, nullptr).
    if (__locale_all != nullptr) {
      __setlocale(__locale_all);
      free(__locale_all);
    }
    _configthreadlocale(__status);
  }
  static const char* __setlocale(const char* __locale) {
    const char* __new_locale = setlocale(LC_ALL, __locale);
    if (__new_locale == nullptr)
      __throw_bad_alloc();
    return __new_locale;
  }
  int __status;
  char* __locale_all = nullptr;
};
#else
struct __locale_guard {
  _LIBCPP_HIDE_FROM_ABI __locale_guard(locale_t& __loc) : __old_loc_(uselocale(__loc)) {}

  _LIBCPP_HIDE_FROM_ABI ~__locale_guard() {
    if (__old_loc_)
      uselocale(__old_loc_);
  }

  locale_t __old_loc_;

  __locale_guard(__locale_guard const&)            = delete;
  __locale_guard& operator=(__locale_guard const&) = delete;
};
#endif

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___LOCALE_DIR_LOCALE_GUARD_H
