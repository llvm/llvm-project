//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// constexpr iterator<false> begin();
// constexpr iterator<true> begin() const
//   requires range<const V> &&
//            regular_invocable<const F&, range_reference_t<const V>>;

#include <ranges>

#include "test_macros.h"
#include "types.h"

template<class T>
concept BeginInvocable = requires(T t) { t.begin(); };

constexpr bool test() {
  int buff[8] = {0, 1, 2, 3, 4, 5, 6, 7};

  {
    std::ranges::transform_view transformView(MoveOnlyView{buff}, PlusOneMutable{});
    assert(transformView.begin().base() == buff);
    assert(*transformView.begin() == 1);
  }

  {
    std::ranges::transform_view transformView(ForwardView{buff}, PlusOneMutable{});
    assert(base(transformView.begin().base()) == buff);
    assert(*transformView.begin() == 1);
  }

  {
    std::ranges::transform_view transformView(InputView{buff}, PlusOneMutable{});
    assert(base(transformView.begin().base()) == buff);
    assert(*transformView.begin() == 1);
  }

  {
    const std::ranges::transform_view transformView(MoveOnlyView{buff}, PlusOne{});
    assert(*transformView.begin() == 1);
  }

  static_assert(!BeginInvocable<const std::ranges::transform_view<MoveOnlyView, PlusOneMutable>>);

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
