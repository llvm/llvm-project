//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr iterator& operator-=(difference_type n);

#include <ranges>
#include <cassert>
#include <concepts>

constexpr bool test() {
  using Iter = std::ranges::iterator_t<std::ranges::repeat_view<int>>;
  std::ranges::repeat_view<int> v(10);
  auto iter = v.begin() + 10;
  iter -= 5;
  assert(iter == v.begin() + 5);

  static_assert(std::same_as<decltype(iter -= 5), Iter&>);

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
