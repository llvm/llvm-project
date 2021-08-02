//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: gcc-10
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// static constexpr size_t size() noexcept;

#include <ranges>
#include <cassert>

#include "test_macros.h"

constexpr bool test() {
  {
    auto sv = std::ranges::single_view<int>(42);
    assert(sv.size() == 1);

    ASSERT_SAME_TYPE(decltype(sv.size()), size_t);
    static_assert(noexcept(sv.size()));
  }
  {
    const auto sv = std::ranges::single_view<int>(42);
    assert(sv.size() == 1);

    ASSERT_SAME_TYPE(decltype(sv.size()), size_t);
    static_assert(noexcept(sv.size()));
  }
  {
    auto sv = std::ranges::single_view<int>(42);
    assert(std::ranges::size(sv) == 1);

    ASSERT_SAME_TYPE(decltype(std::ranges::size(sv)), size_t);
    static_assert(noexcept(std::ranges::size(sv)));
  }
  {
    const auto sv = std::ranges::single_view<int>(42);
    assert(std::ranges::size(sv) == 1);

    ASSERT_SAME_TYPE(decltype(std::ranges::size(sv)), size_t);
    static_assert(noexcept(std::ranges::size(sv)));
  }

  // Test that it's static.
  {
    assert(std::ranges::single_view<int>::size() == 1);

    ASSERT_SAME_TYPE(decltype(std::ranges::single_view<int>::size()), size_t);
    static_assert(noexcept(std::ranges::single_view<int>::size()));
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
