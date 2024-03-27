//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// constexpr auto begin() requires (!simple-view<V>);
// constexpr auto begin() const requires range<const V>;

#include <ranges>
#include <cassert>
#include <utility>

#include "__ranges/concepts.h"
#include "test_macros.h"
#include "test_iterators.h"
#include "test_range.h"
#include "types.h"

struct NonCommonSimpleView : std::ranges::view_base {
  int* begin() const;
  sentinel_wrapper<int*> end() const;
  std::size_t size() { return 0; }  // deliberately non-const
};
static_assert(std::ranges::sized_range<NonCommonSimpleView>);
static_assert(!std::ranges::sized_range<const NonCommonSimpleView>);

constexpr bool test() {
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  // simple-view<V> && sized_range<V> && random_access_range<V>
  {
    static_assert(simple_view<SizedRandomAccessView>);
    static_assert(std::ranges::sized_range<SizedRandomAccessView>);
    static_assert(std::ranges::random_access_range<SizedRandomAccessView>);

    std::ranges::take_view<SizedRandomAccessView> tv(SizedRandomAccessView(buffer), 4);
    assert(tv.begin() == SizedRandomAccessView(buffer).begin());
    ASSERT_SAME_TYPE(decltype(tv.begin()), RandomAccessIter);
  }

  {
    const std::ranges::take_view<SizedRandomAccessView> tv(SizedRandomAccessView(buffer), 4);
    assert(tv.begin() == SizedRandomAccessView(buffer).begin());
    ASSERT_SAME_TYPE(decltype(tv.begin()), RandomAccessIter);
  }

  // simple-view<V> && sized_range<V> && !random_access_range<V>
  {
    static_assert(simple_view<SizedForwardView>);
    static_assert(std::ranges::sized_range<SizedForwardView>);
    static_assert(!std::ranges::random_access_range<SizedForwardView>);

    std::ranges::take_view<SizedForwardView> tv(SizedForwardView{buffer}, 16);        // underlying size is 8
    assert(tv.begin() == std::counted_iterator<ForwardIter>(ForwardIter(buffer), 8)); // expect min(8, 16)
    ASSERT_SAME_TYPE(decltype(tv.begin()), std::counted_iterator<ForwardIter>);
  }

  {
    const std::ranges::take_view<SizedForwardView> tv(SizedForwardView{buffer}, 4);
    assert(tv.begin() == std::counted_iterator<ForwardIter>(ForwardIter(buffer), 4));
    ASSERT_SAME_TYPE(decltype(tv.begin()), std::counted_iterator<ForwardIter>);
  }

  // simple-view<V> && !sized_range<V>
  {
    static_assert(simple_view<MoveOnlyView>);
    static_assert(simple_view<MoveOnlyView>);
    std::ranges::take_view<MoveOnlyView> tv(MoveOnlyView{buffer}, 4);
    assert(tv.begin() == std::counted_iterator<int*>(buffer, 4));
    ASSERT_SAME_TYPE(decltype(tv.begin()), std::counted_iterator<int*>);
  }

  {
    const std::ranges::take_view<MoveOnlyView> tv(MoveOnlyView{buffer}, 4);
    assert(tv.begin() == std::counted_iterator<int*>(buffer, 4));
    ASSERT_SAME_TYPE(decltype(tv.begin()), std::counted_iterator<int*>);
  }

  // simple-view<V> && sized_range<V> && !sized_range<const V>
  {
    static_assert(simple_view<NonCommonSimpleView>);
    static_assert(std::ranges::sized_range<NonCommonSimpleView>);
    static_assert(!std::ranges::sized_range<const NonCommonSimpleView>);

    std::ranges::take_view<NonCommonSimpleView> tv{};
    ASSERT_SAME_TYPE(decltype(tv.begin()), std::counted_iterator<int*>);
    ASSERT_SAME_TYPE(decltype(std::as_const(tv).begin()), std::counted_iterator<int*>);
  }

  // non simple-view<V> && !sized_range<V>
  {
    static_assert(!simple_view<NonSimpleNonSizedView>);
    static_assert(!std::ranges::sized_range<NonSimpleNonSizedView>);

    std::ranges::take_view<NonSimpleNonSizedView> tv{NonSimpleNonSizedView{buffer, buffer + 2}, 4};
    // The count for the counted iterator is the count of the take_view (i.e., 4)
    assert(tv.begin() ==
           std::counted_iterator<common_input_iterator<const int*>>(common_input_iterator<const int*>(buffer), 4));
    ASSERT_SAME_TYPE(decltype(tv.begin()), std::counted_iterator<common_input_iterator<const int*>>);
  }

  // non simple-view<V> && sized_range<V>
  {
    static_assert(!simple_view<NonSimpleSizedView>);
    static_assert(std::ranges::sized_range<NonSimpleSizedView>);

    std::ranges::take_view<NonSimpleSizedView> tv{NonSimpleSizedView{buffer, buffer + 2}, 4};
    // The count for the counted iterator is the min(2, 4) (i.e., 2).
    assert(tv.begin() ==
           std::counted_iterator<common_input_iterator<const int*>>(common_input_iterator<const int*>(buffer), 2));
    ASSERT_SAME_TYPE(decltype(tv.begin()), std::counted_iterator<common_input_iterator<const int*>>);
  }

  // non simple-view<V> && sized_range<V> && random_access_range<V>
  {
    static_assert(!simple_view<NonSimpleSizedRandomView>);
    static_assert(std::ranges::sized_range<NonSimpleSizedRandomView>);
    static_assert(std::ranges::random_access_range<NonSimpleSizedRandomView>);

    std::ranges::take_view<NonSimpleSizedRandomView> tv{NonSimpleSizedRandomView{buffer, buffer + 2}, 4};
    assert(tv.begin() == random_access_iterator<const int*>(buffer));
    ASSERT_SAME_TYPE(decltype(tv.begin()), random_access_iterator<const int*>);
  }
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
