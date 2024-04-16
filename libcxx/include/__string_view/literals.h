//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___STRING_VIEW_LITERALS_H
#define _LIBCPP___STRING_VIEW_LITERALS_H

#include <__config>
#include <__string_view/basic_string_view.h>
#include <cstddef>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 14
inline namespace literals {
inline namespace string_view_literals {
inline _LIBCPP_HIDE_FROM_ABI constexpr basic_string_view<char> operator""sv(const char* __str, size_t __len) noexcept {
  return basic_string_view<char>(__str, __len);
}

#  ifndef _LIBCPP_HAS_NO_WIDE_CHARACTERS
inline _LIBCPP_HIDE_FROM_ABI constexpr basic_string_view<wchar_t>
operator""sv(const wchar_t* __str, size_t __len) noexcept {
  return basic_string_view<wchar_t>(__str, __len);
}
#  endif

#  ifndef _LIBCPP_HAS_NO_CHAR8_T
inline _LIBCPP_HIDE_FROM_ABI constexpr basic_string_view<char8_t>
operator""sv(const char8_t* __str, size_t __len) noexcept {
  return basic_string_view<char8_t>(__str, __len);
}
#  endif

inline _LIBCPP_HIDE_FROM_ABI constexpr basic_string_view<char16_t>
operator""sv(const char16_t* __str, size_t __len) noexcept {
  return basic_string_view<char16_t>(__str, __len);
}

inline _LIBCPP_HIDE_FROM_ABI constexpr basic_string_view<char32_t>
operator""sv(const char32_t* __str, size_t __len) noexcept {
  return basic_string_view<char32_t>(__str, __len);
}
} // namespace string_view_literals
} // namespace literals
#endif

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___STRING_VIEW_LITERALS_H
