// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_STACKTRACE_HASH
#define _LIBCPP_STACKTRACE_HASH

#include <__config>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#if _LIBCPP_STD_VER >= 23

#  include <__functional/hash.h>
#  include <cstddef>
#  include <cstdint>

#  include <__stacktrace/base.h>
#  include <__stacktrace/to_string.h>

_LIBCPP_BEGIN_NAMESPACE_STD

// (19.6.6)
// Hash support [stacktrace.basic.hash]

template <class _Allocator>
struct _LIBCPP_EXPORTED_FROM_ABI hash<basic_stacktrace<_Allocator>> {
  [[nodiscard]] size_t operator()(basic_stacktrace<_Allocator> const& __context) const noexcept {
    size_t __ret = 1;
    for (auto const& __entry : __context.__entries_) {
      __ret += hash<uintptr_t>()(__entry.native_handle());
    }
    return __ret;
  }
};

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 23

_LIBCPP_POP_MACROS

#endif // _LIBCPP_STACKTRACE_HASH
