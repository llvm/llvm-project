//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++17

// UNSUPPORTED: libcpp-has-no-incomplete-pstl

// <algorithm>

// template<class ExecutionPolicy, class ForwardIterator>
//   ForwardIterator min_element(ExecutionPolicy&& exec,
//                               ForwardIterator first, ForwardIterator last);
//
// template<class ExecutionPolicy, class ForwardIterator, class Compare>
//   ForwardIterator min_element(ExecutionPolicy&& exec,
//                               ForwardIterator first, ForwardIterator last,
//                               Compare comp);

#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <string>
#include <vector>

#include "test_macros.h"
#include "test_execution_policies.h"
#include "test_iterators.h"

template <class Iter>
struct Test {
  template <class Policy>
  void operator()(Policy&& policy) {
    { // simple test
      int a[] = {5, 3, 8, 1, 9, 2};
      assert(base(std::min_element(policy, Iter(std::begin(a)), Iter(std::end(a)))) == a + 3);
    }
    { // std::greater
      int a[] = {5, 3, 8, 1, 9, 2};
      assert(base(std::min_element(policy, Iter(std::begin(a)), Iter(std::end(a)), std::greater<>())) == a + 4);
    }
    { // custom comparator
      int a[] = {5, 3, 8, 1, 9, 2};
      assert(base(std::min_element(policy, Iter(std::begin(a)), Iter(std::end(a)), [](int lhs, int rhs) {
               return lhs < rhs;
             })) == a + 3);
    }
    { // empty range
      int a[] = {5, 3, 8, 1, 9, 2};
      assert(base(std::min_element(policy, Iter(std::begin(a)), Iter(std::begin(a)))) == std::begin(a));
    }
    { // single element
      int a[] = {5};
      assert(base(std::min_element(policy, Iter(std::begin(a)), Iter(std::end(a)))) == std::begin(a));
    }
    { // first occurrence is returned when multiple minimums exist
      int a[] = {2, 1, 1, 3, 1, 2};
      assert(base(std::min_element(policy, Iter(std::begin(a)), Iter(std::end(a)))) == a + 1);
    }
    { // min at the beginning
      std::vector<int> v(100, 10);
      v[0] = 1;
      assert(base(std::min_element(policy, Iter(std::data(v)), Iter(std::data(v) + std::size(v)))) == std::data(v));
    }
    { // min at the end
      std::vector<int> v(100, 10);
      v[99] = 1;
      assert(base(std::min_element(policy, Iter(std::data(v)), Iter(std::data(v) + std::size(v)))) ==
             std::data(v) + 99);
    }
    { // check that a large range works
      std::vector<int> v(100000, 10);
      v[50000] = 1;
      assert(base(std::min_element(policy, Iter(std::data(v)), Iter(std::data(v) + std::size(v)))) ==
             std::data(v) + 50000);
    }
    { // check that a size not divisible by SIMD block size works
      std::vector<int> v(49, 10);
      v[48] = 1;
      assert(base(std::min_element(policy, Iter(std::data(v)), Iter(std::data(v) + std::size(v)))) ==
             std::data(v) + 48);
    }
    { // all elements equal - should return first element
      std::vector<int> v(10000, 42);
      assert(base(std::min_element(policy, Iter(std::data(v)), Iter(std::data(v) + std::size(v)))) == std::data(v));
    }
  }
};

int main(int, char**) {
  types::for_each(types::forward_iterator_list<int*>{}, TestIteratorWithPolicies<Test>{});
  types::for_each(types::random_access_iterator_list<int*>{}, TestIteratorWithPolicies<Test>{});

  test_execution_policies([](auto&& policy) {
    { // non-trivially copyable
      std::vector<std::string> v(10000, "c++23");
      v[1001] = "c++17";
      v[2001] = "c++17";
      assert(std::min_element(policy, v.begin(), v.end()) == v.begin() + 1001);
    }
  });

  return 0;
}
