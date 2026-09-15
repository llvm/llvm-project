//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++17

// UNSUPPORTED: libcpp-has-no-incomplete-pstl

// template <class ExecutionPolicy,
//           class BidirectionalIterator>
//   void reverse(ExecutionPolicy&& exec,
//                BidirectionalIterator first,
//                BidirectionalIterator last);

#include <algorithm>
#include <cassert>
#include <functional>
#include <iterator>
#include <limits>
#include <numeric>
#include <type_traits>

#include "test_execution_policies.h"
#include "test_iterators.h"
#include "test_macros.h"
#include "type_algorithms.h"
#include "runway_sample.h"

EXECUTION_POLICY_SFINAE_TEST(reverse);

static_assert(sfinae_test_reverse<int, int*, int*>);
static_assert(!sfinae_test_reverse<std::execution::parallel_policy, int*, int*>);

template <class Iter>
struct Test {
  template <class ExecutionPolicy>
  void operator()(ExecutionPolicy&& policy) {
    { // Check the return type
      static_assert(std::is_same_v<
                    void,
                    decltype(std::reverse(
                        std::declval<std::execution::parallel_policy>(), std::declval<Iter>(), std::declval<Iter>()))>);
    }
    { // Empty range
      int a[] = {0};
      std::reverse(policy, Iter(std::begin(a)), Iter(std::begin(a)));
    }
    { // Single element
      int a[] = {0};
      std::reverse(policy, Iter(std::begin(a)), Iter(std::end(a)));
      assert(a[0] == 0);
    }
    { // Two elements
      int a[] = {0, 1};
      std::reverse(policy, Iter(std::begin(a)), Iter(std::end(a)));
      assert(a[0] == 1);
      assert(a[1] == 0);
    }
    { // Three elements
      int a[] = {0, 1, 2};
      std::reverse(policy, Iter(std::begin(a)), Iter(std::end(a)));
      assert(a[0] == 2);
      assert(a[1] == 1);
      assert(a[2] == 0);
    }
    { // Four elements
      int a[] = {0, 1, 2, 3};
      std::reverse(policy, Iter(std::begin(a)), Iter(std::end(a)));
      assert(a[0] == 3);
      assert(a[1] == 2);
      assert(a[2] == 1);
      assert(a[3] == 0);
    }
    { // Many iotaed elements, reverse up to a sampled position, check and reverse back
      int a[1073];
      std::iota(std::begin(a), std::end(a), 1);
      runway_sample(std::size(a) + 1, [&](size_t i) {
        std::reverse(policy, Iter(std::begin(a)), Iter(std::begin(a) + i));
        for (size_t j = 0; j < i; ++j) {
          assert(a[j] == static_cast<int>(i - j));
        }
        std::reverse(policy, Iter(std::begin(a)), Iter(std::begin(a) + i));
      });
    }
  }
};

int main(int, char**) {
  types::for_each(types::bidirectional_iterator_list<int*>{}, TestIteratorWithPolicies<Test>{});
  return 0;
}
