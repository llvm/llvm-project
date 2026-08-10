//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr const W & operator[](difference_type n) const noexcept;

#include <ranges>
#include <cassert>
#include <concepts>
#include <algorithm>

constexpr bool test() {
  // unbound
  {
    std::ranges::repeat_view<int> v(31);
    auto iter = v.begin();
    assert(std::ranges::all_of(std::views::iota(0, 100), [&v](int i) { return v[i] == 31; }));

    static_assert(noexcept(iter[0]));
    static_assert(std::same_as<decltype(iter[0]), const int&>);
  }

  // bound
  {
    std::ranges::repeat_view<int, int> v(32);
    auto iter = v.begin();
    assert(std::ranges::all_of(v, [](int i) { return i == 32; }));
    static_assert(noexcept(iter[0]));
    static_assert(std::same_as<decltype(iter[0]), const int&>);
  }
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
