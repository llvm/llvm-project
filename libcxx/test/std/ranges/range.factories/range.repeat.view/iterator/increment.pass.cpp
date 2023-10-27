//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr iterator& operator++();
// constexpr void operator++(int);

#include <ranges>
#include <concepts>
#include <cassert>

constexpr bool test() {
  using Iter = std::ranges::iterator_t<std::ranges::repeat_view<int>>;
  std::ranges::repeat_view<int> rv(10);
  using Iter = std::ranges::iterator_t<std::ranges::repeat_view<int>>;
  auto iter  = rv.begin();

  assert(iter++ == rv.begin());
  assert(++iter == rv.begin() + 2);

  static_assert(std::same_as<decltype(iter++), Iter>);
  static_assert(std::same_as<decltype(++iter), Iter&>);

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
