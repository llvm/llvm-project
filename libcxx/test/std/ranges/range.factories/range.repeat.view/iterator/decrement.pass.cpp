//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr iterator& operator--();
// constexpr iterator operator--(int);

#include <ranges>
#include <cassert>

constexpr bool test() {
  using Iter = std::ranges::iterator_t<std::ranges::repeat_view<int>>;
  std::ranges::repeat_view<int> rv(10);
  auto iter = rv.begin() + 10;

  assert(iter-- == rv.begin() + 10);
  assert(--iter == rv.begin() + 8);

  static_assert(std::same_as<decltype(iter--), Iter>);
  static_assert(std::same_as<decltype(--iter), Iter&>);

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
