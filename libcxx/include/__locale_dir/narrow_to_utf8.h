//===-----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___LOCALE_NARROW_TO_UTF8_H
#define _LIBCPP___LOCALE_NARROW_TO_UTF8_H

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
struct __narrow_to_utf8 {
  template <class _OutputIterator, class _CharT>
  _OutputIterator operator()(_OutputIterator __s, const _CharT* __wb, const _CharT* __we) const;
};

template <>
struct __narrow_to_utf8<8> {
  template <class _OutputIterator, class _CharT>
  _LIBCPP_HIDE_FROM_ABI _OutputIterator operator()(_OutputIterator __s, const _CharT* __wb, const _CharT* __we) const {
    for (; __wb < __we; ++__wb, ++__s)
      *__s = *__wb;
    return __s;
  }
};

_LIBCPP_SUPPRESS_DEPRECATED_PUSH
template <>
struct _LIBCPP_EXPORTED_FROM_ABI __narrow_to_utf8<16> : public codecvt<char16_t, char, mbstate_t> {
  _LIBCPP_HIDE_FROM_ABI __narrow_to_utf8() : codecvt<char16_t, char, mbstate_t>(1) {}
  _LIBCPP_SUPPRESS_DEPRECATED_POP

  ~__narrow_to_utf8() override;

  template <class _OutputIterator, class _CharT>
  _LIBCPP_HIDE_FROM_ABI _OutputIterator operator()(_OutputIterator __s, const _CharT* __wb, const _CharT* __we) const {
    result __r = ok;
    mbstate_t __mb;
    while (__wb < __we && __r != error) {
      const int __sz = 32;
      char __buf[__sz];
      char* __bn;
      const char16_t* __wn = (const char16_t*)__wb;
      __r = do_out(__mb, (const char16_t*)__wb, (const char16_t*)__we, __wn, __buf, __buf + __sz, __bn);
      if (__r == codecvt_base::error || __wn == (const char16_t*)__wb)
        __throw_runtime_error("locale not supported");
      for (const char* __p = __buf; __p < __bn; ++__p, ++__s)
        *__s = *__p;
      __wb = (const _CharT*)__wn;
    }
    return __s;
  }
};

_LIBCPP_SUPPRESS_DEPRECATED_PUSH
template <>
struct _LIBCPP_EXPORTED_FROM_ABI __narrow_to_utf8<32> : public codecvt<char32_t, char, mbstate_t> {
  _LIBCPP_HIDE_FROM_ABI __narrow_to_utf8() : codecvt<char32_t, char, mbstate_t>(1) {}
  _LIBCPP_SUPPRESS_DEPRECATED_POP

  ~__narrow_to_utf8() override;

  template <class _OutputIterator, class _CharT>
  _LIBCPP_HIDE_FROM_ABI _OutputIterator operator()(_OutputIterator __s, const _CharT* __wb, const _CharT* __we) const {
    result __r = ok;
    mbstate_t __mb;
    while (__wb < __we && __r != error) {
      const int __sz = 32;
      char __buf[__sz];
      char* __bn;
      const char32_t* __wn = (const char32_t*)__wb;
      __r = do_out(__mb, (const char32_t*)__wb, (const char32_t*)__we, __wn, __buf, __buf + __sz, __bn);
      if (__r == codecvt_base::error || __wn == (const char32_t*)__wb)
        __throw_runtime_error("locale not supported");
      for (const char* __p = __buf; __p < __bn; ++__p, ++__s)
        *__s = *__p;
      __wb = (const _CharT*)__wn;
    }
    return __s;
  }
};

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___LOCALE_NARROW_TO_UTF8_H
