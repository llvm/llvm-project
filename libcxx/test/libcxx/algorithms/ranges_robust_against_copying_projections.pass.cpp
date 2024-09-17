//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <algorithm>

// this test checks that the projections in the ranges algorithms aren't copied/moved

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <deque>
#include <type_traits>

#include "test_macros.h"

struct T {};

struct Proj {
    int *copies_;
    constexpr explicit Proj(int *copies) : copies_(copies) {}
    constexpr Proj(const Proj& rhs) : copies_(rhs.copies_) { *copies_ += 1; }
    constexpr Proj& operator=(const Proj&) = default;
    constexpr void* operator()(T) const { return nullptr; }
};

struct Less {
  constexpr bool operator()(void*, void*) const { return false; }
};

struct Equal {
  constexpr bool operator()(void*, void*) const { return true; }
};

struct UnaryVoid {
  constexpr void operator()(void*) const {}
};

struct UnaryTrue {
  constexpr bool operator()(void*) const { return true; }
};

struct NullaryValue {
  constexpr std::nullptr_t operator()() const { return nullptr; }
};

struct UnaryTransform {
  constexpr T operator()(void*) const { return T(); }
};

struct BinaryTransform {
  constexpr T operator()(void*, void*) const { return T(); }
};

constexpr bool all_the_algorithms()
{
    T a[10] = {};
    T b[10] = {};
    T half[5] = {};
    T *first = a;
    T *mid = a+5;
    T *last = a+10;
    T *first2 = b;
    T *mid2 = b+5;
    T *last2 = b+10;
    void *value = nullptr;
    int count = 1;

    int copies = 0;
    (void)std::ranges::adjacent_find(first, last, Equal(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::adjacent_find(a, Equal(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::all_of(first, last, UnaryTrue(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::all_of(a, UnaryTrue(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::any_of(first, last, UnaryTrue(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::any_of(a, UnaryTrue(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::binary_search(first, last, value, Less(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::binary_search(a, value, Less(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::clamp(T(), T(), T(), Less(), Proj(&copies)); assert(copies == 0);
#if TEST_STD_VER >= 23
    (void)std::ranges::contains(first, last, value, Proj(&copies));
    assert(copies == 0);
    (void)std::ranges::contains(a, value, Proj(&copies));
    assert(copies == 0);
    (void)std::ranges::contains_subrange(first, last, first2, last2, Equal(), Proj(&copies), Proj(&copies));
    assert(copies == 0);
    (void)std::ranges::contains_subrange(a, b, Equal(), Proj(&copies), Proj(&copies));
    assert(copies == 0);
#endif
    (void)std::ranges::count(first, last, value, Proj(&copies)); assert(copies == 0);
    (void)std::ranges::count(a, value, Proj(&copies)); assert(copies == 0);
    (void)std::ranges::count_if(first, last, UnaryTrue(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::count_if(a, UnaryTrue(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::copy_if(first, last, first2, UnaryTrue(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::copy_if(a, first2, UnaryTrue(), Proj(&copies)); assert(copies == 0);
#if TEST_STD_VER >= 23
    (void)std::ranges::ends_with(first, last, first2, last2, Equal(), Proj(&copies), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::ends_with(a, b, Equal(), Proj(&copies), Proj(&copies)); assert(copies == 0);
#endif
    (void)std::ranges::equal(first, last, first2, last2, Equal(), Proj(&copies), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::equal(a, b, Equal(), Proj(&copies), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::equal_range(first, last, value, Less(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::equal_range(a, value, Less(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::find(first, last, value, Proj(&copies)); assert(copies == 0);
    (void)std::ranges::find(a, value, Proj(&copies)); assert(copies == 0);
    (void)std::ranges::find_end(first, last, first2, mid2, Equal(), Proj(&copies), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::find_end(a, b, Equal(), Proj(&copies), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::find_first_of(first, last, first2, last2, Equal(), Proj(&copies), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::find_first_of(a, b, Equal(), Proj(&copies), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::find_if(first, last, UnaryTrue(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::find_if(a, UnaryTrue(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::find_if_not(first, last, UnaryTrue(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::find_if_not(a, UnaryTrue(), Proj(&copies)); assert(copies == 0);
#if TEST_STD_VER >= 23
    (void)std::ranges::find_last(first, last, value, Proj(&copies));
    assert(copies == 0);
    (void)std::ranges::find_last(a, value, Proj(&copies));
    assert(copies == 0);
    (void)std::ranges::find_last_if(first, last, UnaryTrue(), Proj(&copies));
    assert(copies == 0);
    (void)std::ranges::find_last_if(a, UnaryTrue(), Proj(&copies));
    assert(copies == 0);
    (void)std::ranges::find_last_if_not(first, last, UnaryTrue(), Proj(&copies));
    assert(copies == 0);
    (void)std::ranges::find_last_if_not(a, UnaryTrue(), Proj(&copies));
    assert(copies == 0);
#endif
    (void)std::ranges::for_each(first, last, UnaryVoid(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::for_each(a, UnaryVoid(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::for_each_n(first, count, UnaryVoid(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::includes(first, last, first2, last2, Less(), Proj(&copies), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::includes(a, b, Less(), Proj(&copies), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::is_heap(first, last, Less(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::is_heap(a, Less(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::is_heap_until(first, last, Less(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::is_heap_until(a, Less(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::is_partitioned(first, last, UnaryTrue(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::is_partitioned(a, UnaryTrue(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::is_permutation(first, last, first2, last2, Equal(), Proj(&copies), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::is_permutation(a, b, Equal(), Proj(&copies), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::is_sorted(first, last, Less(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::is_sorted(a, Less(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::is_sorted_until(first, last, Less(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::is_sorted_until(a, Less(), Proj(&copies)); assert(copies == 0);
    if (!std::is_constant_evaluated()) { (void)std::ranges::inplace_merge(first, mid, last, Less(), Proj(&copies)); assert(copies == 0); }
    if (!std::is_constant_evaluated()) { (void)std::ranges::inplace_merge(a, mid, Less(), Proj(&copies)); assert(copies == 0); }
    (void)std::ranges::lexicographical_compare(first, last, first2, last2, Less(), Proj(&copies), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::lexicographical_compare(a, b, Less(), Proj(&copies), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::lower_bound(first, last, value, Less(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::lower_bound(a, value, Less(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::make_heap(first, last, Less(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::make_heap(a, Less(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::max(T(), T(), Less(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::max({ T(), T() }, Less(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::max(a, Less(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::max_element(first, last, Less(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::max_element(a, Less(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::merge(first, mid, mid, last, first2, Less(), Proj(&copies), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::merge(half, half, b, Less(), Proj(&copies), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::min(T(), T(), Less(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::min({ T(), T() }, Less(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::min(a, Less(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::min_element(first, last, Less(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::min_element(a, Less(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::minmax(T(), T(), Less(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::minmax({ T(), T() }, Less(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::minmax(a, Less(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::minmax_element(first, last, Less(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::minmax_element(a, Less(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::mismatch(first, last, first2, last2, Equal(), Proj(&copies), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::mismatch(a, b, Equal(), Proj(&copies), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::next_permutation(first, last, Less(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::next_permutation(a, Less(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::none_of(first, last, UnaryTrue(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::none_of(a, UnaryTrue(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::nth_element(first, mid, last, Less(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::nth_element(a, mid, Less(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::partial_sort(first, mid, last, Less(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::partial_sort(a, mid, Less(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::partial_sort_copy(first, last, first2, mid2, Less(), Proj(&copies), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::partial_sort_copy(a, b, Less(), Proj(&copies), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::partition(first, last, UnaryTrue(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::partition(a, UnaryTrue(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::partition_copy(first, last, first2, last2, UnaryTrue(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::partition_copy(a, first2, last2, UnaryTrue(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::partition_point(first, last, UnaryTrue(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::partition_point(a, UnaryTrue(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::pop_heap(first, last, Less(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::pop_heap(a, Less(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::prev_permutation(first, last, Less(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::prev_permutation(a, Less(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::push_heap(first, last, Less(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::push_heap(a, Less(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::remove_copy(first, last, first2, value, Proj(&copies)); assert(copies == 0);
    (void)std::ranges::remove_copy(a, first2, value, Proj(&copies)); assert(copies == 0);
    (void)std::ranges::remove_copy_if(first, last, first2, UnaryTrue(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::remove_copy_if(a, first2, UnaryTrue(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::remove(first, last, value, Proj(&copies)); assert(copies == 0);
    (void)std::ranges::remove(a, value, Proj(&copies)); assert(copies == 0);
    (void)std::ranges::remove_if(first, last, UnaryTrue(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::remove_if(a, UnaryTrue(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::replace_copy(first, last, first2, value, T(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::replace_copy(a, first2, value, T(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::replace_copy_if(first, last, first2, UnaryTrue(), T(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::replace_copy_if(a, first2, UnaryTrue(), T(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::replace(first, last, value, T(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::replace(a, value, T(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::replace_if(first, last, UnaryTrue(), T(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::replace_if(a, UnaryTrue(), T(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::search(first, last, first2, mid2, Equal(), Proj(&copies), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::search(a, b, Equal(), Proj(&copies), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::search_n(first, last, count, value, Equal(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::search_n(a, count, value, Equal(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::set_difference(first, mid, mid, last, first2, Less(), Proj(&copies), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::set_difference(a, b, first2, Less(), Proj(&copies), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::set_intersection(first, mid, mid, last, first2, Less(), Proj(&copies), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::set_intersection(a, b, first2, Less(), Proj(&copies), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::set_symmetric_difference(first, mid, mid, last, first2, Less(), Proj(&copies), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::set_symmetric_difference(a, b, first2, Less(), Proj(&copies), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::set_union(first, mid, mid, last, first2, Less(), Proj(&copies), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::set_union(a, b, first2, Less(), Proj(&copies), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::sort(first, last, Less(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::sort(a, Less(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::sort_heap(first, last, Less(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::sort_heap(a, Less(), Proj(&copies)); assert(copies == 0);
    if (!std::is_constant_evaluated()) { (void)std::ranges::stable_partition(first, last, UnaryTrue(), Proj(&copies)); assert(copies == 0); }
    if (!std::is_constant_evaluated()) { (void)std::ranges::stable_partition(a, UnaryTrue(), Proj(&copies)); assert(copies == 0); }
    if (!std::is_constant_evaluated()) { (void)std::ranges::stable_sort(first, last, Less(), Proj(&copies)); assert(copies == 0); }
    if (!std::is_constant_evaluated()) { (void)std::ranges::stable_sort(a, Less(), Proj(&copies)); assert(copies == 0); }
#if TEST_STD_VER > 20
    (void)std::ranges::starts_with(first, last, first2, last2, Equal(), Proj(&copies), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::starts_with(a, b, Equal(), Proj(&copies), Proj(&copies)); assert(copies == 0);
#endif
    (void)std::ranges::transform(first, last, first2, UnaryTransform(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::transform(a, first2, UnaryTransform(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::transform(first, mid, mid, last, first2, BinaryTransform(), Proj(&copies), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::transform(a, b, first2, BinaryTransform(), Proj(&copies), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::unique(first, last, Equal(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::unique(a, Equal(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::unique_copy(first, last, first2, Equal(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::unique_copy(a, first2, Equal(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::upper_bound(first, last, value, Less(), Proj(&copies)); assert(copies == 0);
    (void)std::ranges::upper_bound(a, value, Less(), Proj(&copies)); assert(copies == 0);

    return true;
}

void test_deque() {
  std::deque<T> d;
  int copies  = 0;
  void* value = nullptr;

  (void)std::ranges::find(d, value, Proj(&copies));
  assert(copies == 0);
}

int main(int, char**) {
  test_deque();
  all_the_algorithms();
  static_assert(all_the_algorithms());

  return 0;
}
