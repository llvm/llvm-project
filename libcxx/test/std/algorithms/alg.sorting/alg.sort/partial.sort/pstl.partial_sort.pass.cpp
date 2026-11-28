//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// REQUIRES: std-at-least-c++17

// UNSUPPORTED: libcpp-has-no-incomplete-pstl

// template<class ExecutionPolicy, class RandomAccessIterator>
//   void partial_sort(ExecutionPolicy&& exec,
//                     RandomAccessIterator first,
//                     RandomAccessIterator middle,
//                     RandomAccessIterator last);

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iterator>
#include <numeric>
#include <type_traits>
#include <random>
#include <vector>

#include "test_execution_policies.h"
#include "test_iterators.h"
#include "test_macros.h"
#include "type_algorithms.h"

EXECUTION_POLICY_SFINAE_TEST(partial_sort);

static_assert(sfinae_test_partial_sort<int, int*, int*, int*>);
static_assert(!sfinae_test_partial_sort<std::execution::parallel_policy, int*, int*, int*>);

// Checks the postconditions of partial_sort: [first, middle) is sorted, no element in [middle, last) is less
// than an element in [first, middle), and [first, last) is a permutation of the original sequence.
void check_partially_sorted(const int* orig, const int* first, const int* middle, const int* last) {
  assert(std::is_sorted(first, middle));
  assert(std::is_permutation(first, last, orig));
  assert(first == middle || std::all_of(middle, last, [&](int x) { return !(x < *(middle - 1)); }));
}

template <class Iter>
struct Test {
  template <class ExecutionPolicy>
  void operator()(ExecutionPolicy&& policy) {
    { // Check the return type
      int a[] = {1};
      static_assert(std::is_same_v<
                    decltype(std::partial_sort(policy, Iter(std::begin(a)), Iter(std::begin(a)), Iter(std::end(a)))),
                    void>);
    }
    { // Check empty range
      std::vector<int> orig = {3, 2, 1};
      std::vector<int> a    = orig;
      std::partial_sort(policy, Iter(a.data()), Iter(a.data()), Iter(a.data()));
      assert(a == orig);
    }
    { // Check empty sorted part
      std::vector<int> orig = {3, 2, 1};
      std::vector<int> a    = orig;
      std::partial_sort(policy, Iter(a.data()), Iter(a.data()), Iter(a.data() + a.size()));
      check_partially_sorted(orig.data(), a.data(), a.data(), a.data() + a.size());
    }
    { // Check one element
      std::vector<int> orig = {1};
      std::vector<int> a    = orig;
      std::partial_sort(policy, Iter(a.data()), Iter(a.data() + a.size()), Iter(a.data() + a.size()));
      check_partially_sorted(orig.data(), a.data(), a.data() + a.size(), a.data() + a.size());
    }
    { // Check the first element only
      std::vector<int> orig = {5, 3, 4, 1, 2};
      std::vector<int> a    = orig;
      std::partial_sort(policy, Iter(a.data()), Iter(a.data() + 1), Iter(a.data() + a.size()));
      check_partially_sorted(orig.data(), a.data(), a.data() + 1, a.data() + a.size());
    }
    { // Check some elements
      std::vector<int> orig = {2, 1, 4, 4, 7, 2, 4, 1, 6};
      std::vector<int> a    = orig;
      std::partial_sort(policy, Iter(a.data()), Iter(a.data() + 4), Iter(a.data() + a.size()));
      check_partially_sorted(orig.data(), a.data(), a.data() + 4, a.data() + a.size());
    }
    { // Check all elements
      std::vector<int> orig = {2, 1, 4, 4, 7, 2, 4, 1, 6};
      std::vector<int> a    = orig;
      std::partial_sort(policy, Iter(a.data()), Iter(a.data() + a.size()), Iter(a.data() + a.size()));
      check_partially_sorted(orig.data(), a.data(), a.data() + a.size(), a.data() + a.size());
    }
    { // Check all but the last element
      std::vector<int> orig = {2, 1, 4, 4, 7, 2, 4, 1, 6};
      std::vector<int> a    = orig;
      std::partial_sort(policy, Iter(a.data()), Iter(a.data() + a.size() - 1), Iter(a.data() + a.size()));
      check_partially_sorted(orig.data(), a.data(), a.data() + a.size() - 1, a.data() + a.size());
    }
    { // Check already sorted
      std::vector<int> orig = {1, 2, 3, 4, 5, 6};
      std::vector<int> a    = orig;
      std::partial_sort(policy, Iter(a.data()), Iter(a.data() + 3), Iter(a.data() + a.size()));
      check_partially_sorted(orig.data(), a.data(), a.data() + 3, a.data() + a.size());
    }
    { // Check reversed
      std::vector<int> orig = {6, 5, 4, 3, 2, 1};
      std::vector<int> a    = orig;
      std::partial_sort(policy, Iter(a.data()), Iter(a.data() + 3), Iter(a.data() + a.size()));
      check_partially_sorted(orig.data(), a.data(), a.data() + 3, a.data() + a.size());
    }
    { // Check all combinations of middle and last
      std::vector<int> orig = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9};
      std::vector<int> a(orig.size());
      for (std::size_t n = 0; n <= orig.size(); ++n) {
        for (std::size_t m = 0; m <= n; ++m) {
          std::copy(orig.begin(), orig.begin() + n, a.begin());
          std::partial_sort(policy, Iter(a.data()), Iter(a.data() + m), Iter(a.data() + n));
          check_partially_sorted(orig.data(), a.data(), a.data() + m, a.data() + n);
        }
      }
    }
    { // Check large input
#ifndef TEST_HAS_NO_RANDOM_DEVICE
      std::vector<int> orig(1000);
      std::iota(std::begin(orig), std::end(orig), 0);
      std::shuffle(std::begin(orig), std::end(orig), std::mt19937_64(std::random_device{}()));
      std::vector<int> v = orig;
      std::partial_sort(policy, Iter(v.data()), Iter(v.data() + 100), Iter(v.data() + v.size()));
      check_partially_sorted(orig.data(), v.data(), v.data() + 100, v.data() + v.size());
#endif
    }
  }
};

int main(int, char**) {
  types::for_each(types::random_access_iterator_list<int*>{}, TestIteratorWithPolicies<Test>{});
  return 0;
}
