//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_STACKTRACE_UTILS_FAILED
#define _LIBCPP_STACKTRACE_UTILS_FAILED

#include <__config>
#include <cerrno>
#include <exception>

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __stacktrace {

struct failed : std::exception {
  char const* msg_{};
  int errno_{0};
  failed(char const* msg, int err) : std::exception(), msg_(msg), errno_(err) {}

  virtual ~failed() noexcept = default;
  const char* what() const noexcept override { return msg_; }
};

} // namespace __stacktrace
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STACKTRACE_UTILS_FAILED
