//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: gcc-10
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// sentinel() = default;
// constexpr explicit sentinel(sentinel_t<Base> end);
// constexpr sentinel(sentinel<!Const> s)
//   requires Const && convertible_to<sentinel_t<V>, sentinel_t<Base>>;

#include <ranges>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"
#include "../types.h"

constexpr bool test() {
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  auto sw = sentinel_wrapper<int *>(buffer + 8); // Note: not 4, but that's OK.

  {
    const std::ranges::take_view<ContiguousView> tv(ContiguousView{buffer}, 4);
    assert(tv.end().base().base() == sw.base());
    ASSERT_SAME_TYPE(decltype(tv.end().base()), sentinel_wrapper<int *>);
  }

  {
    std::ranges::take_view<ContiguousView> tv(ContiguousView{buffer}, 4);
    assert(tv.end().base().base() == sw.base());
    ASSERT_SAME_TYPE(decltype(tv.end().base()), sentinel_wrapper<int *>);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
