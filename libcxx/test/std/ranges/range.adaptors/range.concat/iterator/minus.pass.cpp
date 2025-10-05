//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// friend constexpr iterator operator-(const iterator& it, difference_type n)
//         requires concat-is-random-access<Const, Views...>;

// friend constexpr difference_type operator-(const iterator& x, const iterator& y)
//     requires concat-is-random-access<Const, Views...>;

// friend constexpr difference_type operator-(default_sentinel_t, const __iterator& __x)
//     requires(sized_sentinel_for<sentinel_t<__maybe_const<_Const, _Views>>, iterator_t<__maybe_const<_Const, _Views>>> &&
//              ...) &&
//             (__all_but_first_model_sized_range<_Const, _Views...>::value)

// friend constexpr difference_type operator-(const __iterator& __x, default_sentinel_t)
//     requires(sized_sentinel_for<sentinel_t<__maybe_const<_Const, _Views>>, iterator_t<__maybe_const<_Const, _Views>>> &&
//              ...) &&
//             (__all_but_first_model_sized_range<_Const, _Views...>::value)

#include <array>
#include <cassert>
#include <iterator>
#include <ranges>

#include "test_iterators.h"
#include "test_macros.h"
#include "../types.h"

constexpr bool test() {
  // Test two iterators
  {
    std::array<int, 2> array1{0, 1};
    std::array<int, 2> array2{2, 3};
    std::ranges::concat_view view(std::views::all(array1), std::views::all(array2));
    auto it1 = view.begin();
    it1++;
    it1++;
    auto it2 = view.begin();
    auto res = it1 - it2;
    assert(res == 2);
  }

  // Test one iterator and one sentinel
  {
    std::array<int, 2> array1{0, 1};
    std::array<int, 2> array2{2, 3};
    std::ranges::concat_view view(std::views::all(array1), std::views::all(array2));
    auto it1 = view.begin();
    auto res = std::default_sentinel_t{} - it1;
    assert(res == 4);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
