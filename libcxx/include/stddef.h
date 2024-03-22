// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__need_ptrdiff_t) || defined(__need_size_t) || defined(__need_rsize_t) || defined(__need_wchar_t) ||       \
    defined(__need_NULL) || defined(__need_nullptr_t) || defined(__need_unreachable) || defined(__need_max_align_t) || \
    defined(__need_offsetof) || defined(__need_wint_t)

#  if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#    pragma GCC system_header
#  endif

#  if defined(__need_nullptr_t) && defined(__cplusplus)
// stddef.h will undef __need_nullptr_t
#    define __cxx_need_nullptr_t
#  endif

#  include_next <stddef.h>

#  ifdef __cxx_need_nullptr_t
#    include <__config>
typedef decltype(nullptr) nullptr_t;
#    undef __cxx_need_nullptr_t
#  endif

#elif !defined(_LIBCPP_STDDEF_H) || (defined(__STDC_WANT_LIB_EXT1__) && __STDC_WANT_LIB_EXT1__ >= 1)
#  define _LIBCPP_STDDEF_H

/*
    stddef.h synopsis

Macros:

    offsetof(type,member-designator)
    NULL

Types:

    ptrdiff_t
    size_t
    max_align_t // C++11
    nullptr_t

*/

#  include <__config>

#  if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#    pragma GCC system_header
#  endif

#  if __has_include_next(<stddef.h>)
#    include_next <stddef.h>
#  endif

#  ifdef __cplusplus
typedef decltype(nullptr) nullptr_t;
#  endif

#endif // _LIBCPP_STDDEF_H
