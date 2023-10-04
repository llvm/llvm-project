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
#include "__concepts/same_as.h"
#include "__ranges/stride_view.h"
#include <cassert>
#include <ranges>
#include <type_traits>

constexpr bool non_simple_view_iter_ctor_test() {
  using NotSimpleStrideView     = std::ranges::stride_view<NotSimpleView>;
  using NotSimpleStrideViewIter = std::ranges::iterator_t<NotSimpleStrideView>;

  using SimpleStrideView     = std::ranges::stride_view<ForwardTracedMoveView>;
  using SimpleStrideViewIter = std::ranges::iterator_t<SimpleStrideView>;

  NotSimpleStrideView nsv{NotSimpleView{}, 1};
  [[maybe_unused]] NotSimpleStrideViewIter nsv_iter = nsv.begin();

  SimpleStrideView sv{ForwardTracedMoveView{}, 1};
  [[maybe_unused]] SimpleStrideViewIter ssv_iter = sv.begin();

  using NotSimpleStrideViewIterConst = std::ranges::iterator_t<const NotSimpleStrideView>;
  using SimpleStrideViewIterConst    = std::ranges::iterator_t<const SimpleStrideView>;

  // .begin on a stride view over a non-simple view will give us a
  // stride_view iterator with its _Const == false. Compare that type
  // with an iterator on a stride view over a simple view that will give
  // us an iterator with its _Const == true. They should *not* be the same.
  static_assert(!std::is_same_v<decltype(ssv_iter), decltype(nsv_iter)>);
  static_assert(!std::is_same_v<NotSimpleStrideViewIterConst, decltype(nsv_iter)>);
  static_assert(std::is_same_v<SimpleStrideViewIterConst, decltype(ssv_iter)>);
  return true;
}

int main(int, char**) {
  non_simple_view_iter_ctor_test();
  static_assert(non_simple_view_iter_ctor_test());
  return 0;
}
