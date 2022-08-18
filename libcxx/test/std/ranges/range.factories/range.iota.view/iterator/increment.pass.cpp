//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// constexpr iterator& operator++();
// constexpr void operator++(int);
// constexpr iterator operator++(int) requires incrementable<W>;

#include <ranges>
#include <cassert>

#include "test_macros.h"
#include "../types.h"

constexpr bool test() {
  {
    std::ranges::iota_view<int> io(0);
    auto iter1 = io.begin();
    auto iter2 = io.begin();
    assert(iter1 == iter2);
    assert(++iter1 != iter2++);
    assert(iter1 == iter2);

    static_assert(!std::is_reference_v<decltype(iter2++)>);
    static_assert( std::is_reference_v<decltype(++iter2)>);
    static_assert(std::same_as<std::remove_reference_t<decltype(++iter2)>, decltype(iter2++)>);
  }
  {
    std::ranges::iota_view io(SomeInt(0));
    auto iter1 = io.begin();
    auto iter2 = io.begin();
    assert(iter1 == iter2);
    assert(++iter1 != iter2++);
    assert(iter1 == iter2);

    static_assert(!std::is_reference_v<decltype(iter2++)>);
    static_assert( std::is_reference_v<decltype(++iter2)>);
    static_assert(std::same_as<std::remove_reference_t<decltype(++iter2)>, decltype(iter2++)>);
  }

  {
    std::ranges::iota_view<NotIncrementable> io(NotIncrementable(0));
    auto iter1 = io.begin();
    auto iter2 = io.begin();
    assert(iter1 == iter2);
    assert(++iter1 != iter2);
    iter2++;
    assert(iter1 == iter2);

    static_assert(std::same_as<decltype(iter2++), void>);
    static_assert(std::is_reference_v<decltype(++iter2)>);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
