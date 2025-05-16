// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_BASIC_STACKTRACE_FORMAT
#define _LIBCPP_BASIC_STACKTRACE_FORMAT

#include <__config>
#include <__fwd/format.h>
#include <__memory/allocator_traits.h>
#include <__vector/vector.h>

#include <__stacktrace/base.h>
#include <__stacktrace/to_string.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

// (19.6.5)
// Formatting support [stacktrace.format]

// TODO: stacktrace formatter: https://github.com/llvm/llvm-project/issues/105257
template <class _Allocator>
struct _LIBCPP_EXPORTED_FROM_ABI formatter<basic_stacktrace<_Allocator>>;

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP_BASIC_STACKTRACE_FORMAT
