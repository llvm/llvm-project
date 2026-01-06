// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_FLOAT_H
#define _LIBCPP_FLOAT_H

#include <__config>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#ifdef __cplusplus

// Workround for https://github.com/llvm/llvm-project/issues/172800
#  define __need_next_float_after_libcpp
#  include <float.h>
#  undef __need_next_float_after_libcpp

#  ifndef FLT_EVAL_METHOD
#    define FLT_EVAL_METHOD __FLT_EVAL_METHOD__
#  endif

#  ifndef DECIMAL_DIG
#    define DECIMAL_DIG __DECIMAL_DIG__
#  endif

#endif // __cplusplus

#endif // _LIBCPP_FLOAT_H
