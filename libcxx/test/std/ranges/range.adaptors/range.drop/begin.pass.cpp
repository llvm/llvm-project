//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// constexpr auto begin()
//   requires (!(simple-view<V> &&
//               random_access_range<const V> && sized_range<const V>));
// constexpr auto begin() const
//   requires random_access_range<const V> && sized_range<const V>;

#include <ranges>

#include "test_macros.h"
#include "test_iterators.h"
#include "test_range.h"
#include "types.h"

template<class T>
concept BeginInvocable = requires(std::ranges::drop_view<T> t) { t.begin(); };

template <bool IsSimple>
struct MaybeSimpleView : std::ranges::view_base {
  int* num_of_non_const_begin_calls;
  int* num_of_const_begin_calls;

  constexpr int* begin() {
    ++(*num_of_non_const_begin_calls);
    return nullptr;
  }
  constexpr std::conditional_t<IsSimple, int*, const int*> begin() const {
    ++(*num_of_const_begin_calls);
    return nullptr;
  }
  constexpr int* end() const { return nullptr; }
  constexpr std::size_t size() const { return 0; }
};

using SimpleView = MaybeSimpleView<true>;
using NonSimpleView = MaybeSimpleView<false>;

constexpr bool test() {
  // random_access_range<const V> && sized_range<const V>
  std::ranges::drop_view dropView1(MoveOnlyView(), 4);
  assert(dropView1.begin() == globalBuff + 4);

  // !random_access_range<const V>
  std::ranges::drop_view dropView2(ForwardView(), 4);
  assert(base(dropView2.begin()) == globalBuff + 4);

  // !random_access_range<const V>
  std::ranges::drop_view dropView3(InputView(), 4);
  assert(base(dropView3.begin()) == globalBuff + 4);

  // random_access_range<const V> && sized_range<const V>
  std::ranges::drop_view dropView4(MoveOnlyView(), 8);
  assert(dropView4.begin() == globalBuff + 8);

  // random_access_range<const V> && sized_range<const V>
  std::ranges::drop_view dropView5(MoveOnlyView(), 0);
  assert(dropView5.begin() == globalBuff);

  // random_access_range<const V> && sized_range<const V>
  const std::ranges::drop_view dropView6(MoveOnlyView(), 0);
  assert(dropView6.begin() == globalBuff);

  // random_access_range<const V> && sized_range<const V>
  std::ranges::drop_view dropView7(MoveOnlyView(), 10);
  assert(dropView7.begin() == globalBuff + 8);

  CountedView view8;
  std::ranges::drop_view dropView8(view8, 5);
  assert(base(base(dropView8.begin())) == globalBuff + 5);
  assert(dropView8.begin().stride_count() == 5);
  assert(base(base(dropView8.begin())) == globalBuff + 5);
  assert(dropView8.begin().stride_count() == 5);

  static_assert(!BeginInvocable<const ForwardView>);

  {
    // non-common non-simple view,
    // The wording of the standard is:
    // Returns: ranges::next(ranges::begin(base_), count_, ranges::end(base_))
    // Note that "Returns" is used here, meaning that we don't have to do it this way.
    // In fact, this will use ranges::advance that has O(n) on non-common range.
    // but [range.range] requires "amortized constant time" for ranges::begin and ranges::end
    // Here, we test that begin() is indeed constant time, by creating a customized
    // sentinel and counting how many times the sentinel eq function is called.
    // It should be 0 times, but since this test (or any test under libcxx/test/std) is
    // also used by other implementations, we relax the condition to that
    // sentinel_cmp_calls is a constant number.
    int sentinel_cmp_calls_1 = 0;
    int sentinel_cmp_calls_2 = 0;
    using NonCommonView      = MaybeSimpleNonCommonView<false>;
    static_assert(std::ranges::random_access_range<NonCommonView>);
    static_assert(std::ranges::sized_range<NonCommonView>);
    std::ranges::drop_view dropView9_1(NonCommonView{{}, 0, &sentinel_cmp_calls_1}, 4);
    std::ranges::drop_view dropView9_2(NonCommonView{{}, 0, &sentinel_cmp_calls_2}, 6);
    assert(dropView9_1.begin() == globalBuff + 4);
    assert(dropView9_2.begin() == globalBuff + 6);
    assert(sentinel_cmp_calls_1 == sentinel_cmp_calls_2);
  }

  {
    // non-common simple view, same as above.
    int sentinel_cmp_calls_1 = 0;
    int sentinel_cmp_calls_2 = 0;
    using NonCommonView      = MaybeSimpleNonCommonView<true>;
    static_assert(std::ranges::random_access_range<NonCommonView>);
    static_assert(std::ranges::sized_range<NonCommonView>);
    std::ranges::drop_view dropView10_1(NonCommonView{{}, 0, &sentinel_cmp_calls_1}, 4);
    std::ranges::drop_view dropView10_2(NonCommonView{{}, 0, &sentinel_cmp_calls_2}, 6);
    assert(dropView10_1.begin() == globalBuff + 4);
    assert(dropView10_2.begin() == globalBuff + 6);
    assert(sentinel_cmp_calls_1 == sentinel_cmp_calls_2);
  }

  {
    static_assert(std::ranges::random_access_range<const SimpleView>);
    static_assert(std::ranges::sized_range<const SimpleView>);
    static_assert(simple_view<SimpleView>);
    int non_const_calls = 0;
    int const_calls = 0;
    std::ranges::drop_view dropView(SimpleView{{}, &non_const_calls, &const_calls}, 4);
    assert(dropView.begin() == nullptr);
    assert(non_const_calls == 0);
    assert(const_calls == 1);
    assert(std::as_const(dropView).begin() == nullptr);
    assert(non_const_calls == 0);
    assert(const_calls == 2);
  }

  {
    static_assert(std::ranges::random_access_range<const NonSimpleView>);
    static_assert(std::ranges::sized_range<const NonSimpleView>);
    static_assert(!simple_view<NonSimpleView>);
    int non_const_calls = 0;
    int const_calls = 0;
    std::ranges::drop_view dropView(NonSimpleView{{}, &non_const_calls, &const_calls}, 4);
    assert(dropView.begin() == nullptr);
    assert(non_const_calls == 1);
    assert(const_calls == 0);
    assert(std::as_const(dropView).begin() == nullptr);
    assert(non_const_calls == 1);
    assert(const_calls == 1);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
