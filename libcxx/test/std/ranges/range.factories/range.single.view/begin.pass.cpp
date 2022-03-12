//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// constexpr T* begin() noexcept;
// constexpr const T* begin() const noexcept;

#include <ranges>
#include <cassert>

#include "test_macros.h"

struct Empty {};
struct BigType { char buffer[64] = {10}; };

constexpr bool test() {
  {
    auto sv = std::ranges::single_view<int>(42);
    assert(*sv.begin() == 42);

    ASSERT_SAME_TYPE(decltype(sv.begin()), int*);
    static_assert(noexcept(sv.begin()));
  }
  {
    const auto sv = std::ranges::single_view<int>(42);
    assert(*sv.begin() == 42);

    ASSERT_SAME_TYPE(decltype(sv.begin()), const int*);
    static_assert(noexcept(sv.begin()));
  }

  {
    auto sv = std::ranges::single_view<Empty>(Empty());
    assert(sv.begin() != nullptr);

    ASSERT_SAME_TYPE(decltype(sv.begin()), Empty*);
  }
  {
    const auto sv = std::ranges::single_view<Empty>(Empty());
    assert(sv.begin() != nullptr);

    ASSERT_SAME_TYPE(decltype(sv.begin()), const Empty*);
  }

  {
    auto sv = std::ranges::single_view<BigType>(BigType());
    assert(sv.begin()->buffer[0] == 10);

    ASSERT_SAME_TYPE(decltype(sv.begin()), BigType*);
  }
  {
    const auto sv = std::ranges::single_view<BigType>(BigType());
    assert(sv.begin()->buffer[0] == 10);

    ASSERT_SAME_TYPE(decltype(sv.begin()), const BigType*);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
