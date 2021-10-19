//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
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

  {
    // Test the default ctor.
    std::ranges::take_view<MoveOnlyView> tv(MoveOnlyView{buffer}, 4);
    assert(decltype(tv.end()){} == std::ranges::next(tv.begin(), 4));
  }

  {
    std::ranges::take_view<MoveOnlyView> nonConst(MoveOnlyView{buffer}, 5);
    const std::ranges::take_view<MoveOnlyView> tvConst(MoveOnlyView{buffer}, 5);
    auto sent1 = nonConst.end();
    // Convert to const. Note, we cannot go the other way.
    std::remove_cv_t<decltype(tvConst.end())> sent2 = sent1;

    assert(sent1 == std::ranges::next(tvConst.begin(), 5));
    assert(sent2 == std::ranges::next(tvConst.begin(), 5));
  }

  {
    std::ranges::take_view<CopyableView> tv(CopyableView{buffer}, 6);
    auto sw = sentinel_wrapper<int *>(buffer + 6);
    using Sent = decltype(tv.end());
    Sent sent = Sent(sw);
    assert(sent.base().base() == sw.base());
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
