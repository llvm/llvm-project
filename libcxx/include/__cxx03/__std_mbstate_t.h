// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CXX03___STD_MBSTATE_T_H
#define _LIBCPP___CXX03___STD_MBSTATE_T_H

#include <__cxx03/__config>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

// The goal of this header is to provide std::mbstate_t without requiring all
// of <cuchar> or <cwchar>.

// We define this here to support older versions of glibc <wchar.h> that do
// not define this for clang. This is also set in libc++'s <wchar.h> header,
// and we need to do so here too to avoid a different function signature given
// a different include order.
#ifdef __cplusplus
#  define __CORRECT_ISO_CPP_WCHAR_H_PROTO
#endif

#if defined(_LIBCPP_HAS_MUSL_LIBC)
#  define __NEED_mbstate_t
#  include <bits/alltypes.h>
#  undef __NEED_mbstate_t
#elif __has_include(<bits/types/mbstate_t.h>)
#  include <bits/types/mbstate_t.h> // works on most Unixes
#elif __has_include(<sys/_types/_mbstate_t.h>)
#  include <sys/_types/_mbstate_t.h> // works on Darwin
#elif __has_include(<bits/mbstate_t.h>)
#  include <bits/mbstate_t.h> // works for Android
#elif !defined(_LIBCPP_HAS_NO_WIDE_CHARACTERS) && __has_include_next(<wchar.h>)
#  include_next <wchar.h> // fall back to the C standard provider of mbstate_t
#elif __has_include_next(<uchar.h>)
#  include_next <uchar.h> // <uchar.h> is also required to make mbstate_t visible
#else
#  error "We don't know how to get the definition of mbstate_t without <wchar.h> on your platform."
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

using ::mbstate_t _LIBCPP_USING_IF_EXISTS;

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___CXX03___STD_MBSTATE_T_H
