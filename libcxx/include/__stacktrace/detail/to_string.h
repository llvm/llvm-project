// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_EXPERIMENTAL_STACKTRACE_TO_STRING
#define _LIBCPP_EXPERIMENTAL_STACKTRACE_TO_STRING

#include <__config>
#include <__fwd/sstream.h>
#include <__ostream/basic_ostream.h>
#include <cstddef>

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Allocator>
class _LIBCPP_EXPORTED_FROM_ABI basic_stacktrace;

class _LIBCPP_EXPORTED_FROM_ABI stacktrace_entry;

namespace __stacktrace {

struct _LIBCPP_HIDE_FROM_ABI __to_string {
  _LIBCPP_HIDE_FROM_ABI string operator()(stacktrace_entry const& __entry);

  _LIBCPP_HIDE_FROM_ABI void operator()(ostream& __os, stacktrace_entry const& __entry);

  _LIBCPP_HIDE_FROM_ABI void operator()(ostream& __os, std::stacktrace_entry const* __entries, size_t __count);

  string operator()(std::stacktrace_entry const* __entries, size_t __count);

  template <class _Allocator>
  _LIBCPP_HIDE_FROM_ABI string operator()(basic_stacktrace<_Allocator> const& __st) {
    return (*this)(__st.__entries_.data(), __st.__entries_.size());
  }
};

} // namespace __stacktrace

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_EXPERIMENTAL_STACKTRACE_TO_STRING
