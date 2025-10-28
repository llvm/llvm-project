// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___FILESYSTEM_U8PATH_H
#define _LIBCPP___FILESYSTEM_U8PATH_H

#include <__algorithm/unwrap_iter.h>
#include <__config>
#include <__filesystem/path.h>
#if !defined(_LIBCPP_WIN32API) || _LIBCPP_HAS_LOCALIZATION
#  include <__locale>
#endif
#include <string>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if _LIBCPP_STD_VER >= 17

_LIBCPP_BEGIN_NAMESPACE_FILESYSTEM

#  if defined(_LIBCPP_WIN32API) && _LIBCPP_HAS_LOCALIZATION
#    define _LIBCPP_FS_U8PATH_CONVERTS_ENCODING 1
#  else
#    define _LIBCPP_FS_U8PATH_CONVERTS_ENCODING 0
#  endif

#  if _LIBCPP_FS_U8PATH_CONVERTS_ENCODING
template <class _InputIt, class _Sentinel>
_LIBCPP_HIDE_FROM_ABI string __make_tmp_string_for_u8path(_InputIt __f, _Sentinel __l) {
  static_assert(__is_pathable<_InputIt>::value);
  static_assert(
#    if _LIBCPP_HAS_CHAR8_T
      is_same<typename __is_pathable<_InputIt>::__char_type, char8_t>::value ||
#    endif
      is_same<typename __is_pathable<_InputIt>::__char_type, char>::value);

  if constexpr (is_same_v<_Sentinel, _NullSentinel>) {
    string __tmp;
    constexpr char __sentinel{};
    for (; *__f != __sentinel; ++__f)
      __tmp.push_back(*__f);
    return __tmp;
  } else {
    static_assert(is_same_v<_InputIt, _Sentinel>);
    return string(__f, __l);
  }
}
#  endif // _LIBCPP_FS_U8PATH_CONVERTS_ENCODING

template <class _InputIt, class _Sentinel>
_LIBCPP_HIDE_FROM_ABI path __u8path(_InputIt __f, _Sentinel __l) {
#  if _LIBCPP_FS_U8PATH_CONVERTS_ENCODING
  auto __tmp = std::filesystem::__make_tmp_string_for_u8path(__f, __l);
  using _CVT = __widen_from_utf8<sizeof(wchar_t) * __CHAR_BIT__>;
  std::wstring __w;
  __w.reserve(__tmp.size());
  _CVT()(back_inserter(__w), __tmp.data(), __tmp.data() + __tmp.size());
  return path(__w);
#  else
  return path(__f, __l);
#  endif // _LIBCPP_FS_U8PATH_CONVERTS_ENCODING
}

template <class _InputIt, __enable_if_t<__is_pathable<_InputIt>::value, int> = 0>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_DEPRECATED_WITH_CHAR8_T path u8path(_InputIt __f, _InputIt __l) {
  static_assert(
#  if _LIBCPP_HAS_CHAR8_T
      is_same<typename __is_pathable<_InputIt>::__char_type, char8_t>::value ||
#  endif
          is_same<typename __is_pathable<_InputIt>::__char_type, char>::value,
      "u8path(Iter, Iter) requires Iter have a value_type of type 'char'"
      " or 'char8_t'");
  return std::filesystem::__u8path(__f, __l);
}

template <class _Source, __enable_if_t<__is_pathable<_Source>::value, int> = 0>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_DEPRECATED_WITH_CHAR8_T path u8path(const _Source& __s) {
  static_assert(
#  if _LIBCPP_HAS_CHAR8_T
      is_same<typename __is_pathable<_Source>::__char_type, char8_t>::value ||
#  endif
          is_same<typename __is_pathable<_Source>::__char_type, char>::value,
      "u8path(Source const&) requires Source have a character type of type "
      "'char' or 'char8_t'");
#  if _LIBCPP_FS_U8PATH_CONVERTS_ENCODING
  using _Traits = __is_pathable<_Source>;
  return std::filesystem::__u8path(
      std::__unwrap_iter(_Traits::__range_begin(__s)), std::__unwrap_iter(_Traits::__range_end(__s)));
#  else
  return path(__s);
#  endif // _LIBCPP_FS_U8PATH_CONVERTS_ENCODING
}

#  undef _LIBCPP_FS_U8PATH_CONVERTS_ENCODING

_LIBCPP_END_NAMESPACE_FILESYSTEM

#endif // _LIBCPP_STD_VER >= 17

#endif // _LIBCPP___FILESYSTEM_U8PATH_H
