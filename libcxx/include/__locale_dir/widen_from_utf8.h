//===-----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___LOCALE_WIDEN_FROM_UTF8_H
#define _LIBCPP___LOCALE_WIDEN_FROM_UTF8_H

#include <__config>
#include <__locale_dir/codecvt.h>
#include <__locale_dir/codecvt_base.h>
#include <__std_mbstate_t.h>
#include <cstddef>
#include <stdexcept>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <size_t _Np>
struct __widen_from_utf8 {
  template <class _OutputIterator>
  _OutputIterator operator()(_OutputIterator __s, const char* __nb, const char* __ne) const;
};

template <>
struct __widen_from_utf8<8> {
  template <class _OutputIterator>
  _LIBCPP_HIDE_FROM_ABI _OutputIterator operator()(_OutputIterator __s, const char* __nb, const char* __ne) const {
    for (; __nb < __ne; ++__nb, ++__s)
      *__s = *__nb;
    return __s;
  }
};

_LIBCPP_SUPPRESS_DEPRECATED_PUSH
template <>
struct _LIBCPP_EXPORTED_FROM_ABI __widen_from_utf8<16> : public codecvt<char16_t, char, mbstate_t> {
  _LIBCPP_HIDE_FROM_ABI __widen_from_utf8() : codecvt<char16_t, char, mbstate_t>(1) {}
  _LIBCPP_SUPPRESS_DEPRECATED_POP

  ~__widen_from_utf8() override;

  template <class _OutputIterator>
  _LIBCPP_HIDE_FROM_ABI _OutputIterator operator()(_OutputIterator __s, const char* __nb, const char* __ne) const {
    result __r = ok;
    mbstate_t __mb;
    while (__nb < __ne && __r != error) {
      const int __sz = 32;
      char16_t __buf[__sz];
      char16_t* __bn;
      const char* __nn = __nb;
      __r              = do_in(__mb, __nb, __ne - __nb > __sz ? __nb + __sz : __ne, __nn, __buf, __buf + __sz, __bn);
      if (__r == codecvt_base::error || __nn == __nb)
        __throw_runtime_error("locale not supported");
      for (const char16_t* __p = __buf; __p < __bn; ++__p, ++__s)
        *__s = *__p;
      __nb = __nn;
    }
    return __s;
  }
};

_LIBCPP_SUPPRESS_DEPRECATED_PUSH
template <>
struct _LIBCPP_EXPORTED_FROM_ABI __widen_from_utf8<32> : public codecvt<char32_t, char, mbstate_t> {
  _LIBCPP_HIDE_FROM_ABI __widen_from_utf8() : codecvt<char32_t, char, mbstate_t>(1) {}
  _LIBCPP_SUPPRESS_DEPRECATED_POP

  ~__widen_from_utf8() override;

  template <class _OutputIterator>
  _LIBCPP_HIDE_FROM_ABI _OutputIterator operator()(_OutputIterator __s, const char* __nb, const char* __ne) const {
    result __r = ok;
    mbstate_t __mb;
    while (__nb < __ne && __r != error) {
      const int __sz = 32;
      char32_t __buf[__sz];
      char32_t* __bn;
      const char* __nn = __nb;
      __r              = do_in(__mb, __nb, __ne - __nb > __sz ? __nb + __sz : __ne, __nn, __buf, __buf + __sz, __bn);
      if (__r == codecvt_base::error || __nn == __nb)
        __throw_runtime_error("locale not supported");
      for (const char32_t* __p = __buf; __p < __bn; ++__p, ++__s)
        *__s = *__p;
      __nb = __nn;
    }
    return __s;
  }
};

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___LOCALE_WIDEN_FROM_UTF8_H
