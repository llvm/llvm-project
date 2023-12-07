//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr unreachable_sentinel_t end() const noexcept;
// constexpr iterator end() const requires (!same_as<Bound, unreachable_sentinel_t>);

#include <ranges>
#include <cassert>
#include <concepts>
#include <iterator>

constexpr bool test() {
  // bound
  {
    std::ranges::repeat_view<int, int> rv(0, 10);
    assert(rv.begin() + 10 == rv.end());
    std::same_as<std::ranges::iterator_t<decltype(rv)>> decltype(auto) iter = rv.end();
    static_assert(std::same_as<decltype(*iter), const int&>);
    for (const auto& i : rv) {
      assert(i == 0);
    }
  }

  // unbound
  {
    std::ranges::repeat_view<int> rv(0);
    assert(rv.begin() + 10 != rv.end());
    static_assert(std::same_as<decltype(rv.end()), std::unreachable_sentinel_t>);
    static_assert(noexcept(rv.end()));
    for (const auto& i : rv | std::views::take(10)) {
      assert(i == 0);
    }
  }
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
