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

// constexpr auto begin()
//   requires (!(simple-view<V> &&
//               random_access_range<const V> && sized_range<const V>));
// constexpr auto begin() const
//   requires random_access_range<const V> && sized_range<const V>;

#include <ranges>

#include "test_macros.h"
#include "types.h"

template<class T>
concept BeginInvocable = requires(std::ranges::drop_view<T> t) { t.begin(); };

constexpr bool test() {
  // random_access_range<const V> && sized_range<const V>
  std::ranges::drop_view dropView1(ContiguousView(), 4);
  assert(dropView1.begin() == globalBuff + 4);

  // !random_access_range<const V>
  std::ranges::drop_view dropView2(ForwardView(), 4);
  assert(dropView2.begin().base() == globalBuff + 4);

  // !random_access_range<const V>
  std::ranges::drop_view dropView3(InputView(), 4);
  assert(dropView3.begin().base() == globalBuff + 4);

  // random_access_range<const V> && sized_range<const V>
  std::ranges::drop_view dropView4(ContiguousView(), 8);
  assert(dropView4.begin() == globalBuff + 8);

  // random_access_range<const V> && sized_range<const V>
  std::ranges::drop_view dropView5(ContiguousView(), 0);
  assert(dropView5.begin() == globalBuff);

  // random_access_range<const V> && sized_range<const V>
  const std::ranges::drop_view dropView6(ContiguousView(), 0);
  assert(dropView6.begin() == globalBuff);

  // random_access_range<const V> && sized_range<const V>
  std::ranges::drop_view dropView7(ContiguousView(), 10);
  assert(dropView7.begin() == globalBuff + 8);

  CountedView view8;
  std::ranges::drop_view dropView8(view8, 5);
  assert(dropView8.begin().base().base() == globalBuff + 5);
  assert(dropView8.begin().stride_count() == 5);
  assert(dropView8.begin().base().base() == globalBuff + 5);
  assert(dropView8.begin().stride_count() == 5);

  static_assert(!BeginInvocable<const ForwardView>);

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
