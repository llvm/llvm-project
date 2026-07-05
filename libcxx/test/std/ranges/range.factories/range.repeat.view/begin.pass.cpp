//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr iterator begin() const;

#include <ranges>
#include <cassert>
#include <concepts>

constexpr bool test() {
  // Test unbound && non-const view
  {
    std::ranges::repeat_view<int> rv(0);
    std::same_as<std::ranges::iterator_t<decltype(rv)>> decltype(auto) iter = rv.begin();
    assert(*iter == 0);
  }

  // Test unbound && const view
  {
    const std::ranges::repeat_view<int> rv(0);
    std::same_as<std::ranges::iterator_t<decltype(rv)>> decltype(auto) iter = rv.begin();
    assert(*iter == 0);
  }

  // Test bound && non-const view
  {
    std::ranges::repeat_view<int, int> rv(1024, 10);
    std::same_as<std::ranges::iterator_t<decltype(rv)>> decltype(auto) iter = rv.begin();
    assert(*iter == 1024);
  }

  // Test bound && const view
  {
    const std::ranges::repeat_view<int, long long> rv(1024, 10);
    std::same_as<std::ranges::iterator_t<decltype(rv)>> decltype(auto) iter = rv.begin();
    assert(*iter == 1024);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
