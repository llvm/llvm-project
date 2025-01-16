//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_STACKTRACE_FAILED_H
#define _LIBCPP_STACKTRACE_FAILED_H

#include <stdexcept>

// TODO(stacktrace23) Use std::expected instead?

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __stacktrace {

struct failed : std::runtime_error {
  virtual ~failed() = default;
  int errno_{0};
  failed() : std::runtime_error({}) {}
  failed(char const* msg, int err) : std::runtime_error(msg), errno_(err) {}
};

} // namespace __stacktrace
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STACKTRACE_FAILED_H
