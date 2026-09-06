// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___FORMAT_FMT_CHAR_TYPE_H
#define _LIBCPP___FORMAT_FMT_CHAR_TYPE_H

#include <__concepts/same_as.h>
#include <__config>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if _LIBCPP_STD_VER >= 20

_LIBCPP_BEGIN_NAMESPACE_STD

/// The character type specializations of \ref formatter.
template <class _CharT>
concept __fmt_char_type =
    same_as<_CharT, char>
#  if _LIBCPP_HAS_WIDE_CHARACTERS
    || same_as<_CharT, wchar_t>
#  endif
    ;

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 20

#endif // _LIBCPP___FORMAT_FMT_CHAR_TYPE_H
