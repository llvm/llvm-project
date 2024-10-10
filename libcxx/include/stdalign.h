// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_STDALIGN_H
#define _LIBCPP_STDALIGN_H

/*
    stdalign.h synopsis

Macros:

    __alignas_is_defined
    __alignof_is_defined

*/

#include <__config>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if __has_include_next(<stdalign.h>)
#  include_next <stdalign.h>
#endif

#ifdef __cplusplus
#  undef alignas
#  undef alignof
#  undef __alignas_is_defined
#  undef __alignof_is_defined
#  define __alignas_is_defined 1
#  define __alignof_is_defined 1
#endif

#endif // _LIBCPP_STDALIGN_H
