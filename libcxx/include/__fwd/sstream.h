//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___FWD_SSTREAM_H
#define _LIBCPP___FWD_SSTREAM_H

#include <__config>
#include <__fwd/string.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _CharT, class _Traits = char_traits<_CharT>,
          class _Allocator = allocator<_CharT> >
    class _LIBCPP_TEMPLATE_VIS basic_stringbuf;

template <class _CharT, class _Traits = char_traits<_CharT>,
          class _Allocator = allocator<_CharT> >
    class _LIBCPP_TEMPLATE_VIS basic_istringstream;
template <class _CharT, class _Traits = char_traits<_CharT>,
          class _Allocator = allocator<_CharT> >
    class _LIBCPP_TEMPLATE_VIS basic_ostringstream;
template <class _CharT, class _Traits = char_traits<_CharT>,
          class _Allocator = allocator<_CharT> >
    class _LIBCPP_TEMPLATE_VIS basic_stringstream;

typedef basic_stringbuf<char>        stringbuf;
typedef basic_istringstream<char>    istringstream;
typedef basic_ostringstream<char>    ostringstream;
typedef basic_stringstream<char>     stringstream;

#ifndef _LIBCPP_HAS_NO_WIDE_CHARACTERS
typedef basic_stringbuf<wchar_t>     wstringbuf;
typedef basic_istringstream<wchar_t> wistringstream;
typedef basic_ostringstream<wchar_t> wostringstream;
typedef basic_stringstream<wchar_t>  wstringstream;
#endif

template <class _CharT, class _Traits, class _Allocator>
    class _LIBCPP_PREFERRED_NAME(stringbuf) _LIBCPP_IF_WIDE_CHARACTERS(_LIBCPP_PREFERRED_NAME(wstringbuf)) basic_stringbuf;
template <class _CharT, class _Traits, class _Allocator>
    class _LIBCPP_PREFERRED_NAME(istringstream) _LIBCPP_IF_WIDE_CHARACTERS(_LIBCPP_PREFERRED_NAME(wistringstream)) basic_istringstream;
template <class _CharT, class _Traits, class _Allocator>
    class _LIBCPP_PREFERRED_NAME(ostringstream) _LIBCPP_IF_WIDE_CHARACTERS(_LIBCPP_PREFERRED_NAME(wostringstream)) basic_ostringstream;
template <class _CharT, class _Traits, class _Allocator>
    class _LIBCPP_PREFERRED_NAME(stringstream) _LIBCPP_IF_WIDE_CHARACTERS(_LIBCPP_PREFERRED_NAME(wstringstream)) basic_stringstream;

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___FWD_SSTREAM_H
