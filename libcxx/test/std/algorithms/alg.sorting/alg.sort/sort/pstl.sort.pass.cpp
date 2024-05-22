//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// UNSUPPORTED: libcpp-has-no-incomplete-pstl

// template<class ExecutionPolicy, class RandomAccessIterator>
//   void sort(ExecutionPolicy&& exec,
//             RandomAccessIterator first, RandomAccessIterator last);
//
// template<class ExecutionPolicy, class RandomAccessIterator, class Compare>
//   void sort(ExecutionPolicy&& exec,
//             RandomAccessIterator first, RandomAccessIterator last,
//             Compare comp);

#include <algorithm>
#include <cassert>
#include <numeric>
#include <random>
#include <vector>

#include "test_execution_policies.h"
#include "test_iterators.h"
#include "test_macros.h"

template <class Iter>
struct Test {
  template <class ExecutionPolicy>
  void operator()(ExecutionPolicy&& policy) {
    { // simple test
      int in[]       = {1, 2, 3, 2, 6, 4};
      int expected[] = {1, 2, 2, 3, 4, 6};
      std::sort(policy, Iter(std::begin(in)), Iter(std::end(in)));
      assert(std::equal(std::begin(in), std::end(in), std::begin(expected)));
    }
    { // empty range works
      int in[]       = {1, 2, 3, 2, 6, 4};
      int expected[] = {1, 2, 3, 2, 6, 4};
      std::sort(policy, Iter(std::begin(in)), Iter(std::begin(in)));
      assert(std::equal(std::begin(in), std::end(in), std::begin(expected)));
    }
    { // single element range works
      int in[]       = {1};
      int expected[] = {1};
      std::sort(policy, Iter(std::begin(in)), Iter(std::begin(in)));
      assert(std::equal(std::begin(in), std::end(in), std::begin(expected)));
    }
    { // two element range works
      int in[]       = {2, 1};
      int expected[] = {1, 2};
      std::sort(policy, Iter(std::begin(in)), Iter(std::end(in)));
      assert(std::equal(std::begin(in), std::end(in), std::begin(expected)));
    }
    { // three element range works
      int in[]       = {2, 1, 4};
      int expected[] = {1, 2, 4};
      std::sort(policy, Iter(std::begin(in)), Iter(std::end(in)));
      assert(std::equal(std::begin(in), std::end(in), std::begin(expected)));
    }
    { // longer range works
      int in[]       = {2, 1, 4, 4, 7, 2, 4, 1, 6};
      int expected[] = {1, 1, 2, 2, 4, 4, 4, 6, 7};
      std::sort(policy, Iter(std::begin(in)), Iter(std::end(in)));
      assert(std::equal(std::begin(in), std::end(in), std::begin(expected)));
    }
    { // already sorted
      int in[]       = {1, 1, 2, 2, 4, 4, 4, 6, 7};
      int expected[] = {1, 1, 2, 2, 4, 4, 4, 6, 7};
      std::sort(policy, Iter(std::begin(in)), Iter(std::end(in)));
      assert(std::equal(std::begin(in), std::end(in), std::begin(expected)));
    }
    { // reversed
      int in[]       = {7, 6, 4, 4, 4, 2, 2, 1, 1};
      int expected[] = {1, 1, 2, 2, 4, 4, 4, 6, 7};
      std::sort(policy, Iter(std::begin(in)), Iter(std::end(in)));
      assert(std::equal(std::begin(in), std::end(in), std::begin(expected)));
    }
    { // repeating pattern
      int in[]       = {1, 2, 3, 1, 2, 3, 1, 2, 3};
      int expected[] = {1, 1, 1, 2, 2, 2, 3, 3, 3};
      std::sort(policy, Iter(std::begin(in)), Iter(std::end(in)));
      assert(std::equal(std::begin(in), std::end(in), std::begin(expected)));
    }
    { // a custom comparator is used
      int in[]       = {1, 2, 3, 2, 6, 4};
      int expected[] = {6, 4, 3, 2, 2, 1};
      std::sort(policy, Iter(std::begin(in)), Iter(std::end(in)), std::greater{});
      assert(std::equal(std::begin(in), std::end(in), std::begin(expected)));
    }
#ifndef TEST_HAS_NO_RANDOM_DEVICE
    { // large range
      std::vector<int> vec(300);
      std::iota(std::begin(vec), std::end(vec), 0);
      auto expected = vec;
      std::shuffle(std::begin(vec), std::end(vec), std::mt19937_64(std::random_device{}()));
      std::sort(Iter(std::data(vec)), Iter(std::data(vec) + std::size(vec)));
      assert(vec == expected);
    }
#endif
  }
};

int main(int, char**) {
  types::for_each(types::random_access_iterator_list<int*>{}, TestIteratorWithPolicies<Test>{});

  return 0;
}
