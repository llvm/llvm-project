//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_STACKTRACE_WIN_IMPL_H
#define _LIBCPP_STACKTRACE_WIN_IMPL_H

#include <__config>
#include <__config_site>
#include <cstddef>
#include <cstdlib>
#include <mutex>

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __stacktrace {

struct context;

struct win_impl {
  context& cx_;
  std::lock_guard<std::mutex> guard_;
  win_impl(context& trace);
  ~win_impl();

  void collect(size_t skip, size_t max_depth);
  void ident_modules();
  void symbolize();
  void resolve_lines();
};

} // namespace __stacktrace
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STACKTRACE_WIN_IMPL_H
