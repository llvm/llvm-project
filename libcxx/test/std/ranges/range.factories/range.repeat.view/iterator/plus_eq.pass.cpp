//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr iterator& operator+=(difference_type n);

#include <ranges>
#include <cassert>
#include <concepts>

constexpr bool test() {
  std::ranges::repeat_view<int> v(10);
  using Iter = std::ranges::iterator_t<std::ranges::repeat_view<int>>;
  auto iter1 = v.begin() + 10;
  auto iter2 = v.begin() + 10;
  assert(iter1 == iter2);
  iter1 += 5;
  assert(iter1 != iter2);
  assert(iter1 == iter2 + 5);

  static_assert(std::same_as<decltype(iter2 += 5), Iter&>);
  assert(std::addressof(iter2) == std::addressof(iter2 += 5));

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
