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
#if defined(_LIBCPP_WIN32API)

#  include <cstddef>
#  include <cstdlib>
#  include <mutex>

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __stacktrace {

struct base;

struct win_impl {
  base& base_;

  static std::mutex mutex_;
  static HANDLE proc_;
  static HMODULE exe_;
  static IMAGE_NT_HEADERS* nt_headers_;
  static bool initialized_;
  static HMODULE module_handles_[1024];
  static size_t module_count_; // 0 IFF module enumeration failed

  /*
  The `dbghelp` APIs are not safe to call concurrently (according to their docs)
  so we claim a lock in constructor.
  */
  explicit win_impl(base& base);
  ~win_impl();

  void collect(size_t skip, size_t max_depth);
  void ident_modules();
  void symbolize();
  void resolve_lines();
};

} // namespace __stacktrace
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_WIN32API
#endif // _LIBCPP_STACKTRACE_WIN_IMPL_H
