//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// UNSUPPORTED: libcpp-has-no-incomplete-pstl

// template<class ExecutionPolicy, class ForwardIterator, class Predicate>
//   bool is_partitioned(ExecutionPolicy&& exec,
//                       ForwardIterator first, ForwardIterator last, Predicate pred);

#include <algorithm>
#include <cassert>
#include <vector>

#include "test_iterators.h"
#include "test_execution_policies.h"

template <class Iter>
struct Test {
  template <class Policy>
  void operator()(Policy&& policy) {
    { // simple test
      int a[] = {1, 2, 3, 4, 5};
      assert(std::is_partitioned(policy, Iter(std::begin(a)), Iter(std::end(a)), [](int i) { return i < 3; }));
    }
    { // check that the range is partitioned if the predicate returns true for all elements
      int a[] = {1, 2, 3, 4, 5};
      assert(std::is_partitioned(policy, Iter(std::begin(a)), Iter(std::end(a)), [](int) { return true; }));
    }
    { // check that the range is partitioned if the predicate returns false for all elements
      int a[] = {1, 2, 3, 4, 5};
      assert(std::is_partitioned(policy, Iter(std::begin(a)), Iter(std::end(a)), [](int) { return false; }));
    }
    { // check that false is returned if the range is not partitioned
      int a[] = {1, 2, 3, 2, 5};
      assert(!std::is_partitioned(policy, Iter(std::begin(a)), Iter(std::end(a)), [](int i) { return i < 3; }));
    }
    { // check that an empty range is partitioned
      int a[] = {1, 2, 3, 2, 5};
      assert(std::is_partitioned(policy, Iter(std::begin(a)), Iter(std::begin(a)), [](int i) { return i < 3; }));
    }
    { // check that a single element is partitioned
      int a[] = {1};
      assert(std::is_partitioned(policy, Iter(std::begin(a)), Iter(std::end(a)), [](int i) { return i < 3; }));
    }
    { // check that a range is partitioned when the partition point is the first element
      int a[] = {1, 2, 2, 4, 5};
      assert(std::is_partitioned(policy, Iter(std::begin(a)), Iter(std::end(a)), [](int i) { return i < 2; }));
    }
    { // check that a range is partitioned when the partition point is the last element
      int a[] = {1, 2, 2, 4, 5};
      assert(std::is_partitioned(policy, Iter(std::begin(a)), Iter(std::end(a)), [](int i) { return i < 2; }));
    }
    { // check that a large range works
      std::vector<int> vec(150, 4);
      vec[0] = 2;
      vec[1] = 1;
      assert(std::is_partitioned(policy, Iter(std::data(vec)), Iter(std::data(vec) + std::size(vec)), [](int i) {
        return i < 3;
      }));
    }
  }
};

int main(int, char**) {
  types::for_each(types::forward_iterator_list<int*>{}, TestIteratorWithPolicies<Test>{});

  return 0;
}
