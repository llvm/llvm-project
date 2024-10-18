//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr __iterator& operator+=(difference_type __n)

#include <iterator>
#include <ranges>

#include "../types.h"
#include "__iterator/concepts.h"
#include "__ranges/concepts.h"
#include "test_iterators.h"

template <class T>
concept CanPlusEqual = std::is_same_v<T&, decltype(std::declval<T>() += 1)> && requires(T& t) { t += 1; };

// Make sure that we cannot use += on a stride view iterator
// over an input view.(sized sentinel)
using InputView = BasicTestView<cpp17_input_iterator<int*>, sized_sentinel<cpp17_input_iterator<int*>>>;
using StrideViewOverInputViewIterator = std::ranges::iterator_t<std::ranges::stride_view<InputView>>;

static_assert(std::ranges::input_range<InputView>);
static_assert(!CanPlusEqual<StrideViewOverInputViewIterator>);

// Make sure that we cannot use += on a stride view iterator
// over a forward view.(sized sentinel)
using ForwardView                       = BasicTestView<forward_iterator<int*>, sized_sentinel<forward_iterator<int*>>>;
using StrideViewOverForwardViewIterator = std::ranges::iterator_t<std::ranges::stride_view<ForwardView>>;

static_assert(std::ranges::forward_range<ForwardView>);
static_assert(!CanPlusEqual<StrideViewOverForwardViewIterator>);

// Make sure that we cannot use += on a stride view iterator
// over a bidirectional view. (sized sentinel)
using BidirectionalView = BasicTestView<bidirectional_iterator<int*>, sized_sentinel<bidirectional_iterator<int*>>>;
using StrideViewOverBidirectionalViewIterator = std::ranges::iterator_t<std::ranges::stride_view<BidirectionalView>>;

static_assert(std::ranges::bidirectional_range<BidirectionalView>);
static_assert(!CanPlusEqual<StrideViewOverBidirectionalViewIterator>);

// Make sure that we can use += on a stride view iterator
// over a random access view. (non sized sentinel)
template <typename RandomAccessIterator = random_access_iterator<int*>>
using RandomAccessView                       = BasicTestView<RandomAccessIterator>;
using StrideViewOverRandomAccessViewIterator = std::ranges::iterator_t<std::ranges::stride_view<RandomAccessView<>>>;

static_assert(std::ranges::random_access_range<RandomAccessView<>>);
static_assert(CanPlusEqual<StrideViewOverRandomAccessViewIterator>);

template <std::random_access_iterator Iter, typename Difference>
constexpr bool test_random_access_operator_plus_equal(Iter begin, Iter end, Difference distance) {
  // Test the forward-range operator+= between an iterator and its distance type.
  // Do not use the RandomAccessView defined in types.h to give the test user more power
  // to customize an iterator and a default sentinel.
  using Base = RandomAccessView<Iter>;
  static_assert(std::ranges::random_access_range<Base>);

  auto base_view                  = Base(begin, end);
  auto stride_view_over_base_view = std::ranges::stride_view(base_view, 1);

  auto base_view_offset                  = Base(begin + distance, end);
  auto stride_view_over_base_view_offset = std::ranges::stride_view(base_view_offset, 1);

  auto sv_bv_begin        = stride_view_over_base_view.begin();
  auto sv_bv_offset_begin = stride_view_over_base_view_offset.begin();

  auto sv_bv_begin_after_distance = sv_bv_begin += distance;
  assert(*sv_bv_begin == *sv_bv_offset_begin);
  assert(*sv_bv_begin_after_distance == *sv_bv_offset_begin);

  auto big_step                            = (end - 1) - begin;
  auto stride_view_over_base_view_big_step = std::ranges::stride_view(base_view, big_step);
  sv_bv_begin                              = stride_view_over_base_view_big_step.begin();

  // This += should move us into a position where the __missing_ will come into play.
  // Do a -= 1 here to confirm that the __missing_ is taken into account.
  sv_bv_begin += 2;
  sv_bv_begin -= 1;
  assert(*sv_bv_begin == *(stride_view_over_base_view.begin() + big_step));
  return true;
}

consteval bool do_static_tests() {
  std::vector<int> vec{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  assert(test_random_access_operator_plus_equal(vec.begin(), vec.end(), 4));
  return true;
}

int main(int, char**) {
  constexpr int arr[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  static_assert(test_random_access_operator_plus_equal(arr, arr + 11, 4));
  test_random_access_operator_plus_equal(arr, arr + 11, 4);

  static_assert(do_static_tests());

  std::vector<int> vec{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  assert(test_random_access_operator_plus_equal(vec.begin(), vec.end(), 4));
  return 0;
}
