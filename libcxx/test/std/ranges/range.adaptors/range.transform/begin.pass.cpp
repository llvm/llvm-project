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
    std::ranges::transform_view transformView(ContiguousView{buff}, Increment{});
    assert(transformView.begin().base() == buff);
    assert(*transformView.begin() == 1);
  }

  {
    std::ranges::transform_view transformView(ForwardView{buff}, Increment{});
    assert(transformView.begin().base().base() == buff);
    assert(*transformView.begin() == 1);
  }

  {
    std::ranges::transform_view transformView(InputView{buff}, Increment{});
    assert(transformView.begin().base().base() == buff);
    assert(*transformView.begin() == 1);
  }

  {
    const std::ranges::transform_view transformView(ContiguousView{buff}, IncrementConst{});
    assert(*transformView.begin() == 1);
  }

  static_assert(!BeginInvocable<const std::ranges::transform_view<ContiguousView, Increment>>);

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
