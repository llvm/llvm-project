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

// constexpr T* end() noexcept;
// constexpr const T* end() const noexcept;

#include <ranges>
#include <cassert>

#include "test_macros.h"

struct Empty {};
struct BigType { char buffer[64] = {10}; };

constexpr bool test() {
  {
    auto sv = std::ranges::single_view<int>(42);
    assert(sv.end() == sv.begin() + 1);

    ASSERT_SAME_TYPE(decltype(sv.end()), int*);
    static_assert(noexcept(sv.end()));
  }
  {
    const auto sv = std::ranges::single_view<int>(42);
    assert(sv.end() == sv.begin() + 1);

    ASSERT_SAME_TYPE(decltype(sv.end()), const int*);
    static_assert(noexcept(sv.end()));
  }

  {
    auto sv = std::ranges::single_view<Empty>(Empty());
    assert(sv.end() == sv.begin() + 1);

    ASSERT_SAME_TYPE(decltype(sv.end()), Empty*);
  }
  {
    const auto sv = std::ranges::single_view<Empty>(Empty());
    assert(sv.end() == sv.begin() + 1);

    ASSERT_SAME_TYPE(decltype(sv.end()), const Empty*);
  }

  {
    auto sv = std::ranges::single_view<BigType>(BigType());
    assert(sv.end() == sv.begin() + 1);

    ASSERT_SAME_TYPE(decltype(sv.end()), BigType*);
  }
  {
    const auto sv = std::ranges::single_view<BigType>(BigType());
    assert(sv.end() == sv.begin() + 1);

    ASSERT_SAME_TYPE(decltype(sv.end()), const BigType*);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
