//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: gcc-10, gcc-11
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// Tests that <value_> is a <copyable-box>.

#include <ranges>
#include <cassert>

#include "test_macros.h"

struct NotAssignable {
  NotAssignable() = default;
  NotAssignable(const NotAssignable&) = default;
  NotAssignable(NotAssignable&&) = default;

  NotAssignable& operator=(const NotAssignable&) = delete;
  NotAssignable& operator=(NotAssignable&&) = delete;
};

constexpr bool test() {
  const std::ranges::single_view<NotAssignable> a;
  std::ranges::single_view<NotAssignable> b;
  b = a;
  b = std::move(a);

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
