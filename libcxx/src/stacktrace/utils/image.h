//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_STACKTRACE_UTILS_IMAGE
#define _LIBCPP_STACKTRACE_UTILS_IMAGE

#include <__config>
#include <__stacktrace/base.h>

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __stacktrace {

struct image {
  constexpr static size_t kMaxImages = 256;

  uintptr_t loaded_at_{};
  intptr_t slide_{};
  std::string_view name_{};
  bool is_main_prog_{};

  bool operator<(image const& rhs) const { return loaded_at_ < rhs.loaded_at_; }
  operator bool() const { return !name_.empty(); }
};

} // namespace __stacktrace
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STACKTRACE_UTILS_IMAGE
