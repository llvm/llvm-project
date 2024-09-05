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

#include <cassert>
#include <ranges>
#include <utility>

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

using CommonInputIterPtrConstInt        = common_input_iterator<const int*>;
using CountedCommonInputIterPtrConstInt = std::counted_iterator<CommonInputIterPtrConstInt>;

constexpr bool test() {
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  // simple-view<V> && sized_range<V> && random_access_range<V>
  {
    using ViewTested = SizedRandomAccessView;
    static_assert(simple_view<ViewTested>);
    static_assert(std::ranges::sized_range<ViewTested>);
    static_assert(std::ranges::random_access_range<ViewTested>);

    std::ranges::take_view<ViewTested> tv(ViewTested(buffer), 4);
    assert(tv.begin() == ViewTested(buffer).begin());
    ASSERT_SAME_TYPE(decltype(tv.begin()), RandomAccessIter);

    const std::ranges::take_view<ViewTested> ctv(ViewTested(buffer), 4);
    assert(ctv.begin() == ViewTested(buffer).begin());
    ASSERT_SAME_TYPE(decltype(ctv.begin()), RandomAccessIter);
  }

  // simple-view<V> && sized_range<V> && !random_access_range<V>
  {
    using ViewTested = SizedForwardView;
    static_assert(simple_view<ViewTested>);
    static_assert(std::ranges::sized_range<ViewTested>);
    static_assert(!std::ranges::random_access_range<ViewTested>);

    std::ranges::take_view<ViewTested> tv(ViewTested{buffer}, 16);                    // underlying size is 8
    assert(tv.begin() == std::counted_iterator<ForwardIter>(ForwardIter(buffer), 8)); // expect min(8, 16)
    ASSERT_SAME_TYPE(decltype(tv.begin()), std::counted_iterator<ForwardIter>);

    const std::ranges::take_view<ViewTested> ctv(ViewTested{buffer}, 4);
    assert(ctv.begin() == std::counted_iterator<ForwardIter>(ForwardIter(buffer), 4));
    ASSERT_SAME_TYPE(decltype(ctv.begin()), std::counted_iterator<ForwardIter>);
  }

  // simple-view<V> && !sized_range<V>
  {
    using ViewTested = MoveOnlyView;
    static_assert(simple_view<ViewTested>);
    std::ranges::take_view<ViewTested> tv(ViewTested{buffer}, 4);
    assert(tv.begin() == std::counted_iterator<int*>(buffer, 4));
    ASSERT_SAME_TYPE(decltype(tv.begin()), std::counted_iterator<int*>);

    const std::ranges::take_view<ViewTested> ctv(ViewTested{buffer}, 4);
    assert(ctv.begin() == std::counted_iterator<int*>(buffer, 4));
    ASSERT_SAME_TYPE(decltype(ctv.begin()), std::counted_iterator<int*>);
  }

  // simple-view<V> && sized_range<V> && !sized_range<const V>
  {
    using ViewTested = NonCommonSimpleView;
    static_assert(simple_view<ViewTested>);
    static_assert(std::ranges::sized_range<ViewTested>);
    static_assert(!std::ranges::sized_range<const ViewTested>);

    std::ranges::take_view<ViewTested> tv{};
    ASSERT_SAME_TYPE(decltype(tv.begin()), std::counted_iterator<int*>);
    ASSERT_SAME_TYPE(decltype(std::as_const(tv).begin()), std::counted_iterator<int*>);
  }

  //  !simple-view<V> && !sized_range<V>
  {
    using ViewTested = NonSimpleNonSizedView;
    static_assert(!simple_view<ViewTested>);
    static_assert(!std::ranges::sized_range<ViewTested>);

    std::ranges::take_view<ViewTested> tv{ViewTested{buffer, buffer + 2}, 4};
    // The count for the counted iterator is the count of the take_view (i.e., 4)
    assert(tv.begin() == CountedCommonInputIterPtrConstInt(CommonInputIterPtrConstInt(buffer), 4));
    ASSERT_SAME_TYPE(decltype(tv.begin()), CountedCommonInputIterPtrConstInt);
  }

  // !simple-view<V> && sized_range<V>
  {
    using ViewTested = NonSimpleSizedView;
    static_assert(!simple_view<ViewTested>);
    static_assert(std::ranges::sized_range<ViewTested>);

    std::ranges::take_view<ViewTested> tv{ViewTested{buffer, buffer + 2}, 4};
    // The count for the counted iterator is the min(2, 4) (i.e., 2).
    assert(tv.begin() == CountedCommonInputIterPtrConstInt(CommonInputIterPtrConstInt(buffer), 2));
    ASSERT_SAME_TYPE(decltype(tv.begin()), CountedCommonInputIterPtrConstInt);
  }

  // !simple-view<V> && sized_range<V> && random_access_range<V>
  {
    using ViewTested = NonSimpleSizedRandomView;
    static_assert(!simple_view<ViewTested>);
    static_assert(std::ranges::sized_range<ViewTested>);
    static_assert(std::ranges::random_access_range<ViewTested>);

    std::ranges::take_view<ViewTested> tv{ViewTested{buffer, buffer + 2}, 4};
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
