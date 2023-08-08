//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// ranges

// std::views::stride_view

#include "../test.h"
#include <cassert>
#include <ranges>

bool non_simple_view_iter_ctor_test() {
  using StrideView             = std::ranges::stride_view<NotSimpleView>;
  using StrideViewIterNonConst = std::ranges::iterator_t<StrideView>;
  using StrideViewIterConst    = std::ranges::iterator_t<const StrideView>;

  StrideView sv{NotSimpleView{}, 1};
  StrideViewIterNonConst iter = {sv, sv.base().begin(), 0};
  StrideViewIterConst iterb   = {iter};
  assert(iterb.__end_.moved_from_a == true);
  return true;
}

constexpr bool simpleview_iter_ctor_test() {
  using StrideView     = std::ranges::stride_view<ForwardTracedMoveView>;
  using StrideViewIter = std::ranges::iterator_t<StrideView>;

  StrideView sv{ForwardTracedMoveView{}, 1};
  StrideViewIter iter = {sv, sv.base().begin(), 0};
  // Guarantee that when the iterator is given to the constructor that
  // it is moved there.
  assert(iter.base().moved);

  return true;
}

int main(int, char**) {
  simpleview_iter_ctor_test();
  non_simple_view_iter_ctor_test();
  static_assert(simpleview_iter_ctor_test());
  return 0;
}
