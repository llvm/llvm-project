//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <variant>

// template <class ...Types> class variant;

#include <variant>

#if __has_feature(memory_sanitizer)
#  include <sanitizer/msan_interface.h>
#endif

int main(int, char**) {
#if __has_feature(memory_sanitizer)
  std::variant<double, int> v;
  v.emplace<double>();
  double& d = std::get<double>(v);
  v.emplace<int>();
  if (__msan_test_shadow(&d, sizeof(d)) == -1) {
    // Unexpected: The entire range is accessible.
    return 1;
  }
#endif

  return 0;
}
