// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_STDDEF_H
#define _LIBCPP_STDDEF_H

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

#include <__config>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if __has_include_next(<stddef.h>)
    // The Clang builtin headers only define the types we need when we request it explicitly.
    // TODO: We should fix that in Clang and drop these defines.
#  ifndef __need_ptrdiff_t
#    define __need_ptrdiff_t
#  endif
#  ifndef __need_size_t
#    define __need_size_t
#  endif
#  ifndef __need_wchar_t
#    define __need_wchar_t
#  endif
#  ifndef __need_NULL
#    define __need_NULL
#  endif
#  ifndef __need_STDDEF_H_misc
#    define __need_STDDEF_H_misc
#  endif

#  include_next <stddef.h>

    // Now re-include the header without requesting any specific types, so as to get
    // any other remaining types from stddef.h. This can all go away once the Clang
    // buitin headers stop using these macros.
#  undef __need_ptrdiff_t
#  undef __need_size_t
#  undef __need_wchar_t
#  undef __need_NULL
#  undef __need_STDDEF_H_misc

#  include_next <stddef.h>
#endif

#ifdef __cplusplus
    typedef decltype(nullptr) nullptr_t;
#endif

#endif // _LIBCPP_STDDEF_H
