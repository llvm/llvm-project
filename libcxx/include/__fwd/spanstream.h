//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___FWD_SPANSTREAM_H
#define _LIBCPP___FWD_SPANSTREAM_H

#include <__config>
#include <__fwd/string.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 23

template <class _CharT, class _Traits = char_traits<_CharT>>
class _LIBCPP_TEMPLATE_VIS basic_spanbuf;
template <class _CharT, class _Traits = char_traits<_CharT>>
class _LIBCPP_TEMPLATE_VIS basic_ispanstream;
template <class _CharT, class _Traits = char_traits<_CharT>>
class _LIBCPP_TEMPLATE_VIS basic_ospanstream;
template <class _CharT, class _Traits = char_traits<_CharT>>
class _LIBCPP_TEMPLATE_VIS basic_spanstream;

using spanbuf     = basic_spanbuf<char>;
using ispanstream = basic_ispanstream<char>;
using ospanstream = basic_ospanstream<char>;
using spanstream  = basic_spanstream<char>;

#  ifndef _LIBCPP_HAS_NO_WIDE_CHARACTERS
using wspanbuf     = basic_spanbuf<wchar_t>;
using wispanstream = basic_ispanstream<wchar_t>;
using wospanstream = basic_ospanstream<wchar_t>;
using wspanstream  = basic_spanstream<wchar_t>;
#  endif

#endif // _LIBCPP_STD_VER >= 23

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___FWD_SPANSTREAM_H
