//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// UNSUPPORTED: libcpp-has-no-incomplete-pstl

// <algorithm>

// template<class ExecutionPolicy, class RandomAccessIterator>
//   void stable_sort(ExecutionPolicy&& exec,
//                    RandomAccessIterator first, RandomAccessIterator last);
//
// template<class ExecutionPolicy, class RandomAccessIterator, class Compare>
//   void stable_sort(ExecutionPolicy&& exec,
//                    RandomAccessIterator first, RandomAccessIterator last,
//                    Compare comp);

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <vector>

#include "test_macros.h"
#include "test_execution_policies.h"
#include "test_iterators.h"
#include "MoveOnly.h"

EXECUTION_POLICY_SFINAE_TEST(stable_sort);

static_assert(sfinae_test_stable_sort<int, int*, int*, bool (*)(int)>);
static_assert(!sfinae_test_stable_sort<std::execution::parallel_policy, int*, int*, bool (*)(int, int)>);

struct OrderedValue {
  int value;
  double original_order;
  bool operator==(const OrderedValue& other) const { return other.value == value; }

  auto operator<(const OrderedValue& rhs) const { return value < rhs.value; }
  auto operator>(const OrderedValue& rhs) const { return value > rhs.value; }
};

template <class Iter, std::size_t N, class Policy, class ValueT = typename std::iterator_traits<Iter>::value_type>
void test_one(Policy&& policy, std::array<ValueT, N> input, std::array<ValueT, N> expected) {
  std::stable_sort(policy, Iter(input.data()), Iter(input.data() + input.size()));
  assert(input == expected);
}

template <class Iter>
struct Test {
  template <class Policy>
  void operator()(Policy&& policy) {
    // Empty sequence.
    test_one<Iter, 0>(policy, {}, {});
    // 1-element sequence.
    test_one<Iter, 1>(policy, {1}, {1});
    // 2-element sequence.
    test_one<Iter, 2>(policy, {2, 1}, {1, 2});
    // 3-element sequence.
    test_one<Iter, 3>(policy, {2, 1, 3}, {1, 2, 3});
    // Longer sequence.
    test_one<Iter, 8>(policy, {2, 1, 3, 6, 8, 4, 11, 5}, {1, 2, 3, 4, 5, 6, 8, 11});
    // Longer sequence with duplicates.
    test_one<Iter, 7>(policy, {2, 1, 3, 6, 2, 8, 6}, {1, 2, 2, 3, 6, 6, 8});
    // All elements are the same.
    test_one<Iter, 3>(policy, {1, 1, 1}, {1, 1, 1});
    // Already sorted.
    test_one<Iter, 5>(policy, {1, 2, 3, 4, 5}, {1, 2, 3, 4, 5});
    // Reverse-sorted.
    test_one<Iter, 5>(policy, {5, 4, 3, 2, 1}, {1, 2, 3, 4, 5});
    // Repeating pattern.
    test_one<Iter, 6>(policy, {1, 2, 1, 2, 1, 2}, {1, 1, 1, 2, 2, 2});

    { // The sort is stable (equivalent elements remain in the same order).
      using V        = OrderedValue;
      using Array    = std::array<V, 20>;
      Array in       = {V{10, 10.1}, {12, 12.1}, {3, 3.1},   {5, 5.1}, {3, 3.2}, {3, 3.3}, {11, 11.1},
                        {12, 12.2},  {4, 4.1},   {4, 4.2},   {4, 4.3}, {1, 1.1}, {6, 6.1}, {3, 3.4},
                        {10, 10.2},  {8, 8.1},   {12, 12.3}, {1, 1.2}, {1, 1.3}, {5, 5.2}};
      Array expected = {V{1, 1.1},  {1, 1.2},   {1, 1.3},   {3, 3.1},   {3, 3.2},   {3, 3.3},  {3, 3.4},
                        {4, 4.1},   {4, 4.2},   {4, 4.3},   {5, 5.1},   {5, 5.2},   {6, 6.1},  {8, 8.1},
                        {10, 10.1}, {10, 10.2}, {11, 11.1}, {12, 12.1}, {12, 12.2}, {12, 12.3}};

      std::stable_sort(policy, in.begin(), in.end());
      assert(in == expected);
    }

    { // A custom comparator works and is stable.
      using V     = OrderedValue;
      using Array = std::array<V, 11>;

      Array in = {
          V{1, 1.1},
          {2, 2.1},
          {2, 2.2},
          {3, 3.1},
          {2, 2.3},
          {3, 3.2},
          {4, 4.1},
          {5, 5.1},
          {2, 2.4},
          {5, 5.2},
          {1, 1.2}};
      Array expected = {
          V{5, 5.1},
          {5, 5.2},
          {4, 4.1},
          {3, 3.1},
          {3, 3.2},
          {2, 2.1},
          {2, 2.2},
          {2, 2.3},
          {2, 2.4},
          {1, 1.1},
          {1, 1.2}};

      std::stable_sort(policy, in.begin(), in.end(), std::greater{});
      assert(in == expected);
    }
  }
};

struct NotDefaultConstructible {
  NotDefaultConstructible(int i) : i_(i) {}

  int i_;

  friend bool operator==(NotDefaultConstructible lhs, NotDefaultConstructible rhs) {
    return lhs.i_ == rhs.i_;
  }

  friend bool operator<(NotDefaultConstructible lhs, NotDefaultConstructible rhs) {
    return lhs.i_ < rhs.i_;
  }
};

int main(int, char**) {
  types::for_each(types::concatenate_t<types::random_access_iterator_list<int*>,
                                       types::random_access_iterator_list<MoveOnly*>,
                                       types::random_access_iterator_list<NotDefaultConstructible*>>{},
                  TestIteratorWithPolicies<Test>{});

#ifndef TEST_HAS_NO_EXCEPTIONS
  std::set_terminate(terminate_successful);
  int a[] = {1, 2};
  try {
    std::stable_sort(std::execution::par, std::begin(a), std::end(a), [](int, int) -> bool { throw int{}; });
  } catch (int) {
    assert(false);
  }
#endif

  return 0;
}
