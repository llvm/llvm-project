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
//     constexpr iterator_t<Base> base() const;

#include <cassert>
#include <concepts>
#include <ranges>
#include <vector>

constexpr bool test() {
  std::vector<int> vector = {1, 2, 3, 4, 5, 6};

  std::ranges::chunk_view<std::ranges::ref_view<std::vector<int>>> chunked = vector | std::views::chunk(2);
  auto it                                                                  = chunked.begin();

  std::same_as<std::vector<int>::iterator> decltype(auto) base = it.base();
  assert(base == vector.begin());

  ++it;
  assert(it.base() == vector.begin() + 2);

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
