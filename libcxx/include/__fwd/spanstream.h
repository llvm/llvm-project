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

#if _LIBCPP_STD_VER >= 23

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _CharT, class _Traits = char_traits<_CharT>>
class basic_spanbuf;
template <class _CharT, class _Traits = char_traits<_CharT>>
class basic_ispanstream;
template <class _CharT, class _Traits = char_traits<_CharT>>
class basic_ospanstream;
template <class _CharT, class _Traits = char_traits<_CharT>>
class basic_spanstream;

using spanbuf     = basic_spanbuf<char>;
using ispanstream = basic_ispanstream<char>;
using ospanstream = basic_ospanstream<char>;
using spanstream  = basic_spanstream<char>;

#  if _LIBCPP_HAS_WIDE_CHARACTERS
using wspanbuf     = basic_spanbuf<wchar_t>;
using wispanstream = basic_ispanstream<wchar_t>;
using wospanstream = basic_ospanstream<wchar_t>;
using wspanstream  = basic_spanstream<wchar_t>;
#  endif

template <class _CharT, class _Traits>
class _LIBCPP_PREFERRED_NAME(spanbuf) _LIBCPP_IF_WIDE_CHARACTERS(_LIBCPP_PREFERRED_NAME(wspanbuf)) basic_spanbuf;
template <class _CharT, class _Traits>
class _LIBCPP_PREFERRED_NAME(ispanstream)
    _LIBCPP_IF_WIDE_CHARACTERS(_LIBCPP_PREFERRED_NAME(wispanstream)) basic_ispanstream;
template <class _CharT, class _Traits>
class _LIBCPP_PREFERRED_NAME(ospanstream)
    _LIBCPP_IF_WIDE_CHARACTERS(_LIBCPP_PREFERRED_NAME(wospanstream)) basic_ospanstream;
template <class _CharT, class _Traits>
class _LIBCPP_PREFERRED_NAME(spanstream)
    _LIBCPP_IF_WIDE_CHARACTERS(_LIBCPP_PREFERRED_NAME(wspanstream)) basic_spanstream;

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 23

#endif // _LIBCPP___FWD_SPANSTREAM_H
