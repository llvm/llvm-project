//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// constexpr auto reserve_hint()
//     requires approximately_sized_range<V>
// constexpr auto reserve_hint() const
//     requires approximately_sized_range<const V>

#include <cassert>
#include <ranges>

#include "test_iterators.h"
#include "test_macros.h"
#include "test_range.h"
#include "types.h"

template <class T>
concept ReserveHintEnabled = requires(const std::ranges::take_view<T>& tv) { tv.reserve_hint(); };

constexpr bool test() {
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    static_assert(ReserveHintEnabled<ApproximatelySizedForwardView>);
  }

  {
    std::ranges::take_view<SimpleViewNonSized> tv(SimpleViewNonSized{buffer, buffer + 8}, 100);
    assert(tv.reserve_hint() == 100);
  }

  {
    std::ranges::take_view<ApproximatelySizedForwardView> tv(ApproximatelySizedForwardView{buffer}, 0);
    assert(tv.reserve_hint() == 0);
  }

  {
    const std::ranges::take_view<ApproximatelySizedForwardView> tv(ApproximatelySizedForwardView{buffer}, 2);
    assert(tv.reserve_hint() == 2);
  }

  {
    std::ranges::take_view<ApproximatelySizedForwardView> tv(ApproximatelySizedForwardView{buffer}, 4);
    assert(tv.reserve_hint() == 4);
  }

  {
    const std::ranges::take_view<ApproximatelySizedForwardView> tv(ApproximatelySizedForwardView{buffer}, 6);
    assert(tv.reserve_hint() == 6);
  }

  {
    std::ranges::take_view<ApproximatelySizedForwardView> tv(ApproximatelySizedForwardView{buffer}, 8);
    assert(tv.reserve_hint() == 8);
  }

  {
    const std::ranges::take_view<ApproximatelySizedForwardView> tv(ApproximatelySizedForwardView{buffer}, 8);
    assert(tv.reserve_hint() == 8);
  }

  {
    std::ranges::take_view<ApproximatelySizedForwardView> tv(ApproximatelySizedForwardView{buffer}, 10);
    assert(tv.reserve_hint() == 8);
  }

  {
    const std::ranges::take_view<ApproximatelySizedForwardView> tv(ApproximatelySizedForwardView{buffer}, 10);
    assert(tv.reserve_hint() == 8);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
