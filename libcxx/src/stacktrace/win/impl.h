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

struct builder;

struct win_impl {
  builder& builder_;
  std::lock_guard<std::mutex> guard_;
  static std::mutex mutex_;

  explicit win_impl(builder& builder) : builder_(builder), guard_(mutex_) { global_init(); }

  void global_init();
  void collect(size_t skip, size_t max_depth);
  void ident_modules();
  void symbolize();
  void resolve_lines();

  ~win_impl();
};

} // namespace __stacktrace
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STACKTRACE_WIN_IMPL_H
