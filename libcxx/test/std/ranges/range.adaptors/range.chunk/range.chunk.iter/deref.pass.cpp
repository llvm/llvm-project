//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <ranges>

//   V models forward_range:
//     constexpr value_type iterator::operator*() const;

#include <cassert>
#include <concepts>
#include <ranges>
#include <vector>

constexpr bool test() {
  std::vector<int> vector                                                  = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::ranges::chunk_view<std::ranges::ref_view<std::vector<int>>> chunked = vector | std::views::chunk(3);

  // Test `constexpr value_type iterator::operator*() const`
  {
    using Iterator = std::ranges::iterator_t<decltype(chunked)>;
    static_assert(std::same_as<decltype(*chunked.begin()), typename Iterator::value_type>);

    std::same_as<int&> decltype(auto) v = *(*chunked.begin()).begin();
    assert(v == 1);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
