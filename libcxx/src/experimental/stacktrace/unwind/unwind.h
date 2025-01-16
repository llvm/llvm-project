//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_STACKTRACE_UNWIND_H
#define _LIBCPP_STACKTRACE_UNWIND_H

#include <__config>
#include <__config_site>
#include <cstddef>
#include <cstdlib>

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __stacktrace {

struct _LIBCPP_HIDE_FROM_ABI context;

struct unwind {
  context& cx_;
  void collect(size_t skip, size_t max_depth);
};

} // namespace __stacktrace
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STACKTRACE_UNWIND_H
