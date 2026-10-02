//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr iterator& operator++()
// constexpr void operator++(int)
// constexpr iterator operator++(int)

#include <cassert>
#include <iterator>
#include <ranges>
#include <type_traits>

#include "../types.h"

template <class T>
concept CanPostIncrementVoid = std::is_same_v<void, decltype(std::declval<T>()++)> && requires(T& t) { t++; };
template <class T>
concept CanPostIncrementIterator = std::is_same_v<T, decltype(std::declval<T>()++)> && requires(T& t) { t = t++; };
template <class T>
concept CanPreIncrementIterator = std::is_same_v<T&, decltype(++(std::declval<T>()))> && requires(T& t) { t = ++t; };

// A stride view with a base that is a non forward range returns void from operator++
using InputView = BasicTestView<cpp17_input_iterator<int*>, sized_sentinel<cpp17_input_iterator<int*>>>;
using StrideViewOverInputViewIterator = std::ranges::iterator_t<std::ranges::stride_view<InputView>>;
static_assert(CanPostIncrementVoid<StrideViewOverInputViewIterator>);
static_assert(!CanPostIncrementIterator<StrideViewOverInputViewIterator>);
static_assert(CanPreIncrementIterator<StrideViewOverInputViewIterator>);

// A stride view with a base that is a forward range returns void from operator++
using ForwardView                       = BasicTestView<forward_iterator<int*>, sized_sentinel<forward_iterator<int*>>>;
using StrideViewOverForwardViewIterator = std::ranges::iterator_t<std::ranges::stride_view<ForwardView>>;
static_assert(!CanPostIncrementVoid<StrideViewOverForwardViewIterator>);
static_assert(CanPostIncrementIterator<StrideViewOverForwardViewIterator>);
static_assert(CanPreIncrementIterator<StrideViewOverForwardViewIterator>);

template <typename Iter>
  requires std::sized_sentinel_for<Iter, Iter> && (!std::forward_iterator<Iter>)
constexpr bool test_non_forward_operator_increment(Iter zero_begin, Iter three_begin, Iter end) {
  using Base = BasicTestView<Iter, Iter>;

  auto base        = Base(zero_begin, end);
  auto base_offset = Base(three_begin, end);
  auto sv          = std::ranges::stride_view(base, 3);
  auto sv_offset   = std::ranges::stride_view(base_offset, 3);

  auto it        = sv.begin();
  auto it_offset = sv_offset.begin();

  auto it_after = it; // With a stride of 3, so ++ moves 3 indexes.
  ++it_after;
  assert(*it_offset == *it_after);

  it_after = it;
  it_after++;
  assert(*it_offset == *it_after);

  // See if both get to the end (with pre-increment).
  auto it_to_end = it;
  ++it_to_end; // 3
  ++it_to_end; // 6
  ++it_to_end; // 9
  ++it_to_end; // End

  auto it_offset_to_end = it_offset; // With a stride of 3, so ++ moves 3 indexes.
  ++it_offset_to_end;                // 6
  ++it_offset_to_end;                // 9
  ++it_offset_to_end;                // End

  assert(it_offset_to_end == it_to_end);
  assert(it_offset_to_end == sv_offset.end());
  assert(it_to_end == sv.end());

  // See if both get to the end (with post-increment).
  it_to_end = it;
  it_to_end++; // 3
  it_to_end++; // 6
  it_to_end++; // 9
  it_to_end++; // End

  it_offset_to_end = it_offset; // With a stride of 3, so ++ moves 3 indexes.
  it_offset_to_end++;           // 6
  it_offset_to_end++;           // 9
  it_offset_to_end++;           // End

  assert(it_offset_to_end == it_to_end);
  assert(it_offset_to_end == sv_offset.end());
  assert(it_to_end == sv.end());

  return true;
}

template <std::forward_iterator Iter>
constexpr bool test_forward_operator_increment(Iter begin, Iter end) {
  using Base = BasicTestView<Iter, Iter>;

  using StrideViewIterator = std::ranges::iterator_t<std::ranges::stride_view<Base>>;
  static_assert(std::ranges::forward_range<Base>);
  static_assert(std::weakly_incrementable<StrideViewIterator>);

  auto base = Base(begin, end);
  auto sv   = std::ranges::stride_view(base, 3);
  auto it   = sv.begin();

  // Create a ground truth for comparison.
  auto expected = sv.begin();
  expected++;

  auto it_after = ++it;
  assert(*it_after == *it);
  assert(*it_after == *expected);

  it       = sv.begin();
  it_after = it;
  it_after++;
  assert(*it_after == *expected);

  it             = sv.begin();
  auto it_to_end = it;
  ++it_to_end; // 3
  ++it_to_end; // 6
  ++it_to_end; // 9
  ++it_to_end; // End

  auto it_to_end2 = it;           // With a stride of 3, so ++ moves 3 indexes.
  it_to_end2      = ++it_to_end2; // 3
  it_to_end2      = ++it_to_end2; // 6
  it_to_end2      = ++it_to_end2; // 9
  it_to_end2      = ++it_to_end2; // End

  assert(it_to_end == it_to_end2);
  assert(it_to_end == sv.end());

  it_to_end = it;
  it_to_end++; // 3
  it_to_end++; // 6
  it_to_end++; // 9
  it_to_end++; // End
  assert(it_to_end == sv.end());

  return true;
}

constexpr bool test_properly_handling_missing() {
  // Check whether the "missing" distance to the end gets handled properly.
  using Base   = BasicTestView<int*, int*>;
  int arr[]    = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  auto base    = Base(arr, arr + 10);
  auto strider = std::ranges::stride_view<Base>(base, 7);

  auto strider_iter = strider.end();

  strider_iter--;
  assert(*strider_iter == 8);

  // Now that we are back among the valid, we should
  // have a normal stride length back (i.e., there is no
  // gap between the last stride and the end).
  strider_iter--;
  assert(*strider_iter == 1);

  strider_iter++;
  assert(*strider_iter == 8);

  // By striding past the end, we are going to generate
  // another gap between the last stride and the end.
  strider_iter++;
  assert(strider_iter == strider.end());

  // Make sure that all sentinel operations work!
  assert(strider.end() == std::default_sentinel);
  assert(std::default_sentinel == strider.end());

  assert(strider_iter - std::default_sentinel == 0);
  assert(std::default_sentinel - strider.end() == 0);
  assert(std::default_sentinel - strider_iter == 0);

  // Let's make sure that the newly regenerated gap gets used.
  strider_iter += -2;
  assert(*strider_iter == 1);

  return true;
}

constexpr bool test() {
  {
    int arr[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    test_forward_operator_increment(arr, arr + 11);
  }
  test_properly_handling_missing();
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  // Non-forward iterators can't be tested in a constexpr context.
  {
    int arr[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    test_non_forward_operator_increment(SizedInputIter(arr), SizedInputIter(arr + 3), SizedInputIter(arr + 10));
  }

  return 0;
}
