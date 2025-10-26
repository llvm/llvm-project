//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

#include <algorithm>
#include <cassert>
#include <compare>
#include <cstddef>
#include <deque>
#include <ranges>
#include <type_traits>
#include <vector>

#include "test_macros.h"

template <class T>
struct Less {
  int* copies_;
  TEST_CONSTEXPR explicit Less(int* copies) : copies_(copies) {}
  TEST_CONSTEXPR_CXX14 Less(const Less& rhs) : copies_(rhs.copies_) { *copies_ += 1; }
  TEST_CONSTEXPR_CXX14 Less& operator=(const Less&) = default;
  TEST_CONSTEXPR bool operator()(T, T) const { return false; }
};

template <class T>
struct Equal {
  int* copies_;
  TEST_CONSTEXPR explicit Equal(int* copies) : copies_(copies) {}
  TEST_CONSTEXPR_CXX14 Equal(const Equal& rhs) : copies_(rhs.copies_) { *copies_ += 1; }
  TEST_CONSTEXPR_CXX14 Equal& operator=(const Equal&) = default;
  TEST_CONSTEXPR bool operator()(T, T) const { return true; }
};

template <class T>
struct UnaryVoid {
  int* copies_;
  TEST_CONSTEXPR explicit UnaryVoid(int* copies) : copies_(copies) {}
  TEST_CONSTEXPR_CXX14 UnaryVoid(const UnaryVoid& rhs) : copies_(rhs.copies_) { *copies_ += 1; }
  TEST_CONSTEXPR_CXX14 UnaryVoid& operator=(const UnaryVoid&) = default;
  TEST_CONSTEXPR_CXX14 void operator()(T) const {}
};

template <class T>
struct UnaryTrue {
  int* copies_;
  TEST_CONSTEXPR explicit UnaryTrue(int* copies) : copies_(copies) {}
  TEST_CONSTEXPR_CXX14 UnaryTrue(const UnaryTrue& rhs) : copies_(rhs.copies_) { *copies_ += 1; }
  TEST_CONSTEXPR_CXX14 UnaryTrue& operator=(const UnaryTrue&) = default;
  TEST_CONSTEXPR bool operator()(T) const { return true; }
};

template <class T>
struct NullaryValue {
  int* copies_;
  TEST_CONSTEXPR explicit NullaryValue(int* copies) : copies_(copies) {}
  TEST_CONSTEXPR_CXX14 NullaryValue(const NullaryValue& rhs) : copies_(rhs.copies_) { *copies_ += 1; }
  TEST_CONSTEXPR_CXX14 NullaryValue& operator=(const NullaryValue&) = default;
  TEST_CONSTEXPR T operator()() const { return 0; }
};

template <class T>
struct UnaryTransform {
  int* copies_;
  TEST_CONSTEXPR explicit UnaryTransform(int* copies) : copies_(copies) {}
  TEST_CONSTEXPR_CXX14 UnaryTransform(const UnaryTransform& rhs) : copies_(rhs.copies_) { *copies_ += 1; }
  TEST_CONSTEXPR_CXX14 UnaryTransform& operator=(const UnaryTransform&) = default;
  TEST_CONSTEXPR T operator()(T) const { return 0; }
};

template <class T>
struct BinaryTransform {
  int* copies_;
  TEST_CONSTEXPR explicit BinaryTransform(int* copies) : copies_(copies) {}
  TEST_CONSTEXPR_CXX14 BinaryTransform(const BinaryTransform& rhs) : copies_(rhs.copies_) { *copies_ += 1; }
  TEST_CONSTEXPR_CXX14 BinaryTransform& operator=(const BinaryTransform&) = default;
  TEST_CONSTEXPR T operator()(T, T) const { return 0; }
};

#if TEST_STD_VER > 17
template <class T>
struct ThreeWay {
  int* copies_;
  constexpr explicit ThreeWay(int* copies) : copies_(copies) {}
  constexpr ThreeWay(const ThreeWay& rhs) : copies_(rhs.copies_) { *copies_ += 1; }
  constexpr ThreeWay& operator=(const ThreeWay&) = default;
  constexpr std::strong_ordering operator()(T, T) const { return std::strong_ordering::equal; }
};
#endif

template <class T>
TEST_CONSTEXPR_CXX20 bool all_the_algorithms() {
  T a[10]   = {};
  T b[10]   = {};
  T* first  = a;
  T* mid    = a + 5;
  T* last   = a + 10;
  T* first2 = b;
  T* mid2   = b + 5;
  T* last2  = b + 10;
  T value   = 0;
  int count = 1;

  int copies = 0;
  (void)std::adjacent_find(first, last, Equal<T>(&copies));
  assert(copies == 0);
#if TEST_STD_VER >= 11
  (void)std::all_of(first, last, UnaryTrue<T>(&copies));
  assert(copies == 0);
  (void)std::any_of(first, last, UnaryTrue<T>(&copies));
  assert(copies == 0);
#endif
  (void)std::binary_search(first, last, value, Less<T>(&copies));
  assert(copies == 0);
#if TEST_STD_VER > 17
  (void)std::clamp(value, value, value, Less<T>(&copies));
  assert(copies == 0);
#endif
  (void)std::count_if(first, last, UnaryTrue<T>(&copies));
  assert(copies == 0);
  (void)std::copy_if(first, last, first2, UnaryTrue<T>(&copies));
  assert(copies == 0);
  (void)std::equal(first, last, first2, Equal<T>(&copies));
  assert(copies == 0);
#if TEST_STD_VER > 11
  (void)std::equal(first, last, first2, last2, Equal<T>(&copies));
  assert(copies == 0);
#endif
  (void)std::equal_range(first, last, value, Less<T>(&copies));
  assert(copies == 0);
  (void)std::find_end(first, last, first2, mid2, Equal<T>(&copies));
  assert(copies == 0);
  (void)std::find_first_of(first, last, first2, last2, Equal<T>(&copies));
  assert(copies == 0);
  (void)std::find_if(first, last, UnaryTrue<T>(&copies));
  assert(copies == 0);
  (void)std::find_if_not(first, last, UnaryTrue<T>(&copies));
  assert(copies == 0);
  (void)std::for_each(first, last, UnaryVoid<T>(&copies));
  assert(copies == 1);
  copies = 0;
#if TEST_STD_VER > 14
  (void)std::for_each_n(first, count, UnaryVoid<T>(&copies));
  assert(copies == 0);
#endif
  (void)std::generate(first, last, NullaryValue<T>(&copies));
  assert(copies == 0);
  (void)std::generate_n(first, count, NullaryValue<T>(&copies));
  assert(copies == 0);
  (void)std::includes(first, last, first2, last2, Less<T>(&copies));
  assert(copies == 0);
  (void)std::is_heap(first, last, Less<T>(&copies));
  assert(copies == 0);
  (void)std::is_heap_until(first, last, Less<T>(&copies));
  assert(copies == 0);
  (void)std::is_partitioned(first, last, UnaryTrue<T>(&copies));
  assert(copies == 0);
  (void)std::is_permutation(first, last, first2, Equal<T>(&copies));
  assert(copies == 0);
#if TEST_STD_VER > 11
  (void)std::is_permutation(first, last, first2, last2, Equal<T>(&copies));
  assert(copies == 0);
#endif
  (void)std::is_sorted(first, last, Less<T>(&copies));
  assert(copies == 0);
  (void)std::is_sorted_until(first, last, Less<T>(&copies));
  assert(copies == 0);
  if (!TEST_IS_CONSTANT_EVALUATED) {
    (void)std::inplace_merge(first, mid, last, Less<T>(&copies));
    assert(copies == 0);
  }
  (void)std::lexicographical_compare(first, last, first2, last2, Less<T>(&copies));
  assert(copies == 0);
#if TEST_STD_VER > 17
  (void)std::lexicographical_compare_three_way(first, last, first2, last2, ThreeWay<T>(&copies));
  assert(copies == 0);
#endif
  (void)std::lower_bound(first, last, value, Less<T>(&copies));
  assert(copies == 0);
  (void)std::make_heap(first, last, Less<T>(&copies));
  assert(copies == 0);
  (void)std::max(value, value, Less<T>(&copies));
  assert(copies == 0);
#if TEST_STD_VER >= 11
  (void)std::max({value, value}, Less<T>(&copies));
  assert(copies == 0);
#endif
  (void)std::max_element(first, last, Less<T>(&copies));
  assert(copies == 0);
  (void)std::merge(first, mid, mid, last, first2, Less<T>(&copies));
  assert(copies == 0);
  (void)std::min(value, value, Less<T>(&copies));
  assert(copies == 0);
#if TEST_STD_VER >= 11
  (void)std::min({value, value}, Less<T>(&copies));
  assert(copies == 0);
#endif
  (void)std::min_element(first, last, Less<T>(&copies));
  assert(copies == 0);
  (void)std::minmax(value, value, Less<T>(&copies));
  assert(copies == 0);
#if TEST_STD_VER >= 11
  (void)std::minmax({value, value}, Less<T>(&copies));
  assert(copies == 0);
#endif
  (void)std::minmax_element(first, last, Less<T>(&copies));
  assert(copies == 0);
  (void)std::mismatch(first, last, first2, Equal<T>(&copies));
  assert(copies == 0);
#if TEST_STD_VER > 11
  (void)std::mismatch(first, last, first2, last2, Equal<T>(&copies));
  assert(copies == 0);
#endif
  (void)std::next_permutation(first, last, Less<T>(&copies));
  assert(copies == 0);
#if TEST_STD_VER >= 11
  (void)std::none_of(first, last, UnaryTrue<T>(&copies));
  assert(copies == 0);
#endif
  (void)std::nth_element(first, mid, last, Less<T>(&copies));
  assert(copies == 0);
  (void)std::partial_sort(first, mid, last, Less<T>(&copies));
  assert(copies == 0);
  (void)std::partial_sort_copy(first, last, first2, mid2, Less<T>(&copies));
  assert(copies == 0);
  (void)std::partition(first, last, UnaryTrue<T>(&copies));
  assert(copies == 0);
  (void)std::partition_copy(first, last, first2, last2, UnaryTrue<T>(&copies));
  assert(copies == 0);
  (void)std::partition_point(first, last, UnaryTrue<T>(&copies));
  assert(copies == 0);
  (void)std::pop_heap(first, last, Less<T>(&copies));
  assert(copies == 0);
  (void)std::prev_permutation(first, last, Less<T>(&copies));
  assert(copies == 0);
  (void)std::push_heap(first, last, Less<T>(&copies));
  assert(copies == 0);
  (void)std::remove_copy_if(first, last, first2, UnaryTrue<T>(&copies));
  assert(copies == 0);
  (void)std::remove_if(first, last, UnaryTrue<T>(&copies));
  assert(copies == 0);
  (void)std::replace_copy_if(first, last, first2, UnaryTrue<T>(&copies), value);
  assert(copies == 0);
  (void)std::replace_if(first, last, UnaryTrue<T>(&copies), value);
  assert(copies == 0);
  (void)std::search(first, last, first2, mid2, Equal<T>(&copies));
  assert(copies == 0);
  (void)std::search_n(first, last, count, value, Equal<T>(&copies));
  assert(copies == 0);
  (void)std::set_difference(first, mid, mid, last, first2, Less<T>(&copies));
  assert(copies == 0);
  (void)std::set_intersection(first, mid, mid, last, first2, Less<T>(&copies));
  assert(copies == 0);
  (void)std::set_symmetric_difference(first, mid, mid, last, first2, Less<T>(&copies));
  assert(copies == 0);
  (void)std::set_union(first, mid, mid, last, first2, Less<T>(&copies));
  assert(copies == 0);
  (void)std::sort(first, first + 3, Less<T>(&copies));
  assert(copies == 0);
  (void)std::sort(first, first + 4, Less<T>(&copies));
  assert(copies == 0);
  (void)std::sort(first, first + 5, Less<T>(&copies));
  assert(copies == 0);
  (void)std::sort(first, last, Less<T>(&copies));
  assert(copies == 0);
  (void)std::sort_heap(first, last, Less<T>(&copies));
  assert(copies == 0);
  if (!TEST_IS_CONSTANT_EVALUATED) {
    (void)std::stable_partition(first, last, UnaryTrue<T>(&copies));
    assert(copies == 0);
  }
  if (!TEST_IS_CONSTANT_EVALUATED) {
    (void)std::stable_sort(first, last, Less<T>(&copies));
    assert(copies == 0);
  }
  (void)std::transform(first, last, first2, UnaryTransform<T>(&copies));
  assert(copies == 0);
  (void)std::transform(first, mid, mid, first2, BinaryTransform<T>(&copies));
  assert(copies == 0);
  (void)std::unique(first, last, Equal<T>(&copies));
  assert(copies == 0);
  (void)std::unique_copy(first, last, first2, Equal<T>(&copies));
  assert(copies == 0);
  (void)std::upper_bound(first, last, value, Less<T>(&copies));
  assert(copies == 0);

  return true;
}

bool test_segmented_iterator() {
  int copies = 0;
  std::deque<int> dq(10);
  (void)std::for_each(dq.begin(), dq.end(), UnaryVoid<int>(&copies));
  assert(copies == 1);
  copies = 0;

#if TEST_STD_VER >= 20
  std::vector<std::vector<int>> vecs(3, std::vector<int>(10));
  auto v = std::views::join(vecs);
  (void)std::for_each(v.begin(), v.end(), UnaryVoid<int>(&copies));
  assert(copies == 1);
  copies = 0;
#endif

  return true;
}

int main(int, char**) {
  all_the_algorithms<void*>();
  all_the_algorithms<int>();
  assert(test_segmented_iterator());
#if TEST_STD_VER > 17
  static_assert(all_the_algorithms<void*>());
  static_assert(all_the_algorithms<int>());
#endif

  return 0;
}
