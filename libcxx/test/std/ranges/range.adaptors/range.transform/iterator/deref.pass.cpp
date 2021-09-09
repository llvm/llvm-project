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

// transform_view::<iterator>::operator*

#include <ranges>

#include "test_macros.h"
#include "../types.h"

int main(int, char**) {
  {
    int buff[] = {0, 1, 2, 3, 4, 5, 6, 7};
    using View = std::ranges::transform_view<ContiguousView, PlusOne>;
    View transformView(ContiguousView{buff}, PlusOne{});
    assert(*transformView.begin() == 1);
    static_assert(!noexcept(*std::declval<std::ranges::iterator_t<View>>()));
    ASSERT_SAME_TYPE(int, decltype(*std::declval<View>().begin()));
  }
  {
    int buff[] = {0, 1, 2, 3, 4, 5, 6, 7};
    using View = std::ranges::transform_view<ContiguousView, PlusOneMutable>;
    View transformView(ContiguousView{buff}, PlusOneMutable{});
    assert(*transformView.begin() == 1);
    static_assert(!noexcept(*std::declval<std::ranges::iterator_t<View>>()));
    ASSERT_SAME_TYPE(int, decltype(*std::declval<View>().begin()));
  }
  {
    int buff[] = {0, 1, 2, 3, 4, 5, 6, 7};
    using View = std::ranges::transform_view<ContiguousView, PlusOneNoexcept>;
    View transformView(ContiguousView{buff}, PlusOneNoexcept{});
    assert(*transformView.begin() == 1);
    static_assert(noexcept(*std::declval<std::ranges::iterator_t<View>>()));
    ASSERT_SAME_TYPE(int, decltype(*std::declval<View>().begin()));
  }
  {
    int buff[] = {0, 1, 2, 3, 4, 5, 6, 7};
    using View = std::ranges::transform_view<ContiguousView, Increment>;
    View transformView(ContiguousView{buff}, Increment{});
    assert(*transformView.begin() == 1);
    static_assert(!noexcept(*std::declval<std::ranges::iterator_t<View>>()));
    ASSERT_SAME_TYPE(int&, decltype(*std::declval<View>().begin()));
  }
  {
    int buff[] = {0, 1, 2, 3, 4, 5, 6, 7};
    using View = std::ranges::transform_view<ContiguousView, IncrementRvalueRef>;
    View transformView(ContiguousView{buff}, IncrementRvalueRef{});
    assert(*transformView.begin() == 1);
    static_assert(!noexcept(*std::declval<std::ranges::iterator_t<View>>()));
    ASSERT_SAME_TYPE(int&&, decltype(*std::declval<View>().begin()));
  }

  return 0;
}
