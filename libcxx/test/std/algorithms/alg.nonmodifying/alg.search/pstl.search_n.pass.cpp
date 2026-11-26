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

// template<class ExecutionPolicy, class ForwardIterator, class Size, class T>
//   ForwardIterator
//   search_n(ExecutionPolicy&& exec,
//            ForwardIterator first, ForwardIterator last,
//            Size count, const T& value);

#include <algorithm>
#include <cassert>
#include <iterator>

#include "test_execution_policies.h"
#include "test_iterators.h"
#include "test_macros.h"
#include "type_algorithms.h"
#include "runway_sample.h"

EXECUTION_POLICY_SFINAE_TEST(search_n);

static_assert(sfinae_test_search_n<int, int*, int*, int, int>);
static_assert(!sfinae_test_search_n<std::execution::parallel_policy, int*, int*, int, int>);

template <class Iter>
struct Test {
  template <class ExecutionPolicy>
  void operator()(ExecutionPolicy&& policy) {
    { // check the return types
      int a[]  = {0};
      auto res = std::search_n(policy, Iter(std::begin(a)), Iter(std::begin(a)), 1, 0);
      static_assert(std::is_same_v<decltype(res), Iter>);
    }
    { // single element range with count = 1, matching
      int a[]  = {5};
      auto ret = std::search_n(policy, Iter(std::begin(a)), Iter(std::end(a)), 1, 5);
      assert(ret == Iter(std::begin(a)));
    }
    { // single element range with count = 1, not matching
      int a[]  = {5};
      auto ret = std::search_n(policy, Iter(std::begin(a)), Iter(std::end(a)), 1, 3);
      assert(ret == Iter(std::end(a)));
    }
    { // simple test - single match in the middle, count = 1
      int a[]  = {1, 2, 3, 4, 5, 6};
      auto ret = std::search_n(policy, Iter(std::begin(a)), Iter(std::end(a)), 1, 3);
      assert(ret == Iter(std::begin(a) + 2));
    }
    { // matching part begins at the front
      int a[]  = {7, 7, 3, 7, 3, 6};
      auto ret = std::search_n(policy, Iter(std::begin(a)), Iter(std::end(a)), 2, 7);
      assert(ret == Iter(std::begin(a)));
    }
    { // matching part ends at the back
      int a[]  = {9, 3, 6, 4, 4};
      auto ret = std::search_n(policy, Iter(std::begin(a)), Iter(std::end(a)), 2, 4);
      assert(ret == Iter(std::begin(a) + 3));
    }
    { // pattern does not match
      int a[]  = {9, 3, 6, 4, 8};
      auto ret = std::search_n(policy, Iter(std::begin(a)), Iter(std::end(a)), 1, 1);
      assert(ret == Iter(std::end(a)));
    }
    { // range and pattern are identical
      int a[]  = {1, 1, 1, 1};
      auto ret = std::search_n(policy, Iter(std::begin(a)), Iter(std::end(a)), 4, 1);
      assert(ret == Iter(std::begin(a)));
    }
    { // pattern is longer than range
      int a[]  = {5};
      auto ret = std::search_n(policy, Iter(std::begin(a)), Iter(std::end(a)), 2, 5);
      assert(ret == Iter(std::end(a)));
    }
    { // pattern is longer than range
      int a[]  = {3, 3, 3};
      auto ret = std::search_n(policy, Iter(std::begin(a)), Iter(std::end(a)), 4, 3);
      assert(ret == Iter(std::end(a)));
    }
    { // pattern has zero length
      int a[]  = {6, 7, 8};
      auto ret = std::search_n(policy, Iter(std::begin(a)), Iter(std::end(a)), 0, 7);
      assert(ret == Iter(std::begin(a)));
    }
    { // range has zero length
      int a[]  = {0};
      auto ret = std::search_n(policy, Iter(std::begin(a)), Iter(std::begin(a)), 1, 1);
      assert(ret == Iter(std::begin(a)));
    }
    {   // check that the first match is returned
      { // match is at the start
        int a[]  = {6, 6, 8, 6, 6, 8, 6, 6, 8};
        auto ret = std::search_n(policy, Iter(std::begin(a)), Iter(std::end(a)), 2, 6);
        assert(ret == Iter(std::begin(a)));
      }
      { // match is in the middle
        int a[]  = {6, 8, 8, 6, 6, 8, 6, 6, 8};
        auto ret = std::search_n(policy, Iter(std::begin(a)), Iter(std::end(a)), 2, 6);
        assert(ret == Iter(std::begin(a) + 3));
      }
      { // match is at the end
        int a[]  = {6, 6, 8, 6, 6, 8, 6, 6, 6};
        auto ret = std::search_n(policy, Iter(std::begin(a)), Iter(std::end(a)), 3, 6);
        assert(ret == Iter(std::begin(a) + 6));
      }
      { // multiple overlapping potential matches
        int a[]  = {3, 3, 3, 3, 3};
        auto ret = std::search_n(policy, Iter(std::begin(a)), Iter(std::end(a)), 3, 3);
        assert(ret == Iter(std::begin(a)));
      }
    }
    { // first almost matches, second is a match
      int a[]  = {6, 6, 6, 7, 6, 6, 6, 6};
      auto ret = std::search_n(policy, Iter(std::begin(a)), Iter(std::end(a)), 4, 6);
      assert(ret == Iter(std::begin(a) + 4));
    }
    { // big haystack, small needle (size=1)
      int a[1073];
      std::fill(std::begin(a), std::end(a), 0);
      runway_sample(std::size(a), [&](size_t i) {
        a[i]     = 1;
        auto ret = std::search_n(policy, Iter(std::begin(a)), Iter(std::end(a)), 1, 1);
        assert(ret == Iter(std::begin(a) + i));
        a[i] = 0;
      });
    }
    { // big haystack, medium needle (size=73)
      int a[1073];
      constexpr size_t needle_size = 73;
      std::fill(std::begin(a), std::end(a), 0);
      runway_sample(std::size(a) - needle_size + 1, [&](size_t i) {
        std::fill(std::begin(a) + i, std::begin(a) + i + needle_size, 1);
        auto ret = std::search_n(policy, Iter(std::begin(a)), Iter(std::end(a)), needle_size, 1);
        assert(ret == Iter(std::begin(a) + i));
        std::fill(std::begin(a) + i, std::begin(a) + i + needle_size, 0);
      });
    }
    { // big haystack, same size needle
      int a[1073];
      std::fill(std::begin(a), std::end(a), 1);
      auto ret = std::search_n(policy, Iter(std::begin(a)), Iter(std::end(a)), std::size(a), 1);
      assert(ret == Iter(std::begin(a)));
    }
  }
};

int main(int, char**) {
  types::for_each(types::forward_iterator_list<const int*>{}, TestIteratorWithPolicies<Test>{});
  return 0;
}
