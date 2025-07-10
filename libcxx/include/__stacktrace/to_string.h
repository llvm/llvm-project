// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_STACKTRACE_TO_STRING
#define _LIBCPP_STACKTRACE_TO_STRING

#include <__config>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#if _LIBCPP_STD_VER >= 23

#  include <__fwd/ostream.h>
#  include <string>

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Allocator>
class basic_stacktrace;

class stacktrace_entry;

namespace __stacktrace {

struct __to_string {
  _LIBCPP_EXPORTED_FROM_ABI string operator()(stacktrace_entry const& __entry);

  _LIBCPP_EXPORTED_FROM_ABI void operator()(ostream& __os, stacktrace_entry const& __entry);

  _LIBCPP_EXPORTED_FROM_ABI void operator()(ostream& __os, std::stacktrace_entry const* __entries, size_t __count);

  _LIBCPP_EXPORTED_FROM_ABI string operator()(std::stacktrace_entry const* __entries, size_t __count);

  template <class _Allocator>
  _LIBCPP_EXPORTED_FROM_ABI string operator()(basic_stacktrace<_Allocator> const& __st) {
    return (*this)(__st.__entries_.data(), __st.__entries_.size());
  }
};

} // namespace __stacktrace

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 23

_LIBCPP_POP_MACROS

#endif // _LIBCPP_STACKTRACE_TO_STRING
