// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_BASIC_STACKTRACE_NONMEM
#define _LIBCPP_BASIC_STACKTRACE_NONMEM

#include <__config>
#if _LIBCPP_STD_VER >= 23

#  include <__memory/allocator_traits.h>
#  include <__utility/swap.h>
#  include <__vector/vector.h>
#  include <string>

#  include <__stacktrace/base.h>
#  include <__stacktrace/to_string.h>

#  if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#    pragma GCC system_header
#  endif

_LIBCPP_PUSH_MACROS
#  include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

// (19.6.4.6)
// Non-member functions [stacktrace.basic.nonmem]

template <class _Allocator>
_LIBCPP_EXPORTED_FROM_ABI inline void
swap(basic_stacktrace<_Allocator>& __a, basic_stacktrace<_Allocator>& __b) noexcept(noexcept(__a.swap(__b))) {
  __a.swap(__b);
}

template <class _Allocator>
_LIBCPP_EXPORTED_FROM_ABI inline string to_string(const basic_stacktrace<_Allocator>& __stacktrace) {
  return __stacktrace::__to_string()(__stacktrace);
}

template <class _Allocator>
_LIBCPP_EXPORTED_FROM_ABI inline ostream& operator<<(ostream& __os, const basic_stacktrace<_Allocator>& __stacktrace) {
  auto __str = __stacktrace::__to_string()(__stacktrace);
  return __os << __str;
}

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP_STD_VER >= 23
#endif // _LIBCPP_BASIC_STACKTRACE_NONMEM
