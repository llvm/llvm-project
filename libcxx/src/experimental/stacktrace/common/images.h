//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_STACKTRACE_IMAGES_H
#define _LIBCPP_STACKTRACE_IMAGES_H

#include <cstdint>
#include <string_view>

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __stacktrace {

constexpr unsigned k_max_images = 256;

struct image {
  uintptr_t loaded_at_{};
  intptr_t slide_{};
  std::string_view name_{};
  bool is_main_prog_{};

  bool operator<(image const& rhs) const { return loaded_at_ < rhs.loaded_at_; }
  operator bool() const { return !name_.empty(); }
};

} // namespace __stacktrace
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STACKTRACE_IMAGES_H
