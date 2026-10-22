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

// template <class ExecutionPolicy,
//           class BidirectionalIterator,
//           class Predicate>
//   BidirectionalIterator stable_partition(ExecutionPolicy&& exec,
//                                          BidirectionalIterator first,
//                                          BidirectionalIterator last,
//                                          Predicate pred);

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

EXECUTION_POLICY_SFINAE_TEST(stable_partition);

static_assert(sfinae_test_stable_partition<int, int*, int*, bool (*)(int, int)>);
static_assert(!sfinae_test_stable_partition<std::execution::parallel_policy, int*, int*, bool (*)(int, int)>);

template <class Iter>
struct Test {
  template <class ExecutionPolicy>
  void operator()(ExecutionPolicy&& policy) {
    using P     = std::pair<int, int>;
    auto is_odd = [](const P& p) { return p.first % 2 == 1; };
    { // Check the return type
      P a[]  = {P(0, 1)};
      auto r = std::stable_partition(policy, Iter(std::begin(a)), Iter(std::end(a)), is_odd);
      static_assert(std::is_same_v<decltype(r), Iter>);
    }
    { // Check mixed
      P a[]  = {P(0, 1), P(0, 2), P(1, 1), P(1, 2), P(2, 1), P(2, 2), P(3, 1), P(3, 2), P(4, 1), P(4, 2)};
      auto r = std::stable_partition(policy, Iter(std::begin(a)), Iter(std::end(a)), is_odd);
      assert(r == Iter(std::begin(a) + 4));
      assert(a[0] == P(1, 1));
      assert(a[1] == P(1, 2));
      assert(a[2] == P(3, 1));
      assert(a[3] == P(3, 2));
      assert(a[4] == P(0, 1));
      assert(a[5] == P(0, 2));
      assert(a[6] == P(2, 1));
      assert(a[7] == P(2, 2));
      assert(a[8] == P(4, 1));
      assert(a[9] == P(4, 2));
    }
    { // Check empty
      P a[]  = {P(0, 1)};
      auto r = std::stable_partition(policy, Iter(std::begin(a)), Iter(std::begin(a)), is_odd);
      assert(r == Iter(std::begin(a)));
    }
    { // Check one true
      P a[]  = {P(1, 1)};
      auto r = std::stable_partition(policy, Iter(std::begin(a)), Iter(std::end(a)), is_odd);
      assert(r == Iter(std::end(a)));
      assert(a[0] == P(1, 1));
    }
    { // Check one false
      P a[]  = {P(0, 1)};
      auto r = std::stable_partition(policy, Iter(std::begin(a)), Iter(std::end(a)), is_odd);
      assert(r == Iter(std::begin(a)));
      assert(a[0] == P(0, 1));
    }
    { // Check all false
      P a[]  = {P(0, 1), P(0, 2), P(2, 1), P(2, 2), P(4, 1), P(4, 2), P(6, 1), P(6, 2), P(8, 1), P(8, 2)};
      auto r = std::stable_partition(policy, Iter(std::begin(a)), Iter(std::end(a)), is_odd);
      assert(r == Iter(std::begin(a)));
      assert(a[0] == P(0, 1));
      assert(a[1] == P(0, 2));
      assert(a[2] == P(2, 1));
      assert(a[3] == P(2, 2));
      assert(a[4] == P(4, 1));
      assert(a[5] == P(4, 2));
      assert(a[6] == P(6, 1));
      assert(a[7] == P(6, 2));
      assert(a[8] == P(8, 1));
      assert(a[9] == P(8, 2));
    }
    { // Check all true
      P a[]  = {P(1, 1), P(1, 2), P(3, 1), P(3, 2), P(5, 1), P(5, 2), P(7, 1), P(7, 2), P(9, 1), P(9, 2)};
      auto r = std::stable_partition(policy, Iter(std::begin(a)), Iter(std::end(a)), is_odd);
      assert(r == Iter(std::end(a)));
      assert(a[0] == P(1, 1));
      assert(a[1] == P(1, 2));
      assert(a[2] == P(3, 1));
      assert(a[3] == P(3, 2));
      assert(a[4] == P(5, 1));
      assert(a[5] == P(5, 2));
      assert(a[6] == P(7, 1));
      assert(a[7] == P(7, 2));
      assert(a[8] == P(9, 1));
      assert(a[9] == P(9, 2));
    }
    { // Check all false but first true
      P a[]  = {P(1, 1), P(0, 2), P(2, 1), P(2, 2), P(4, 1), P(4, 2), P(6, 1), P(6, 2), P(8, 1), P(8, 2)};
      auto r = std::stable_partition(policy, Iter(std::begin(a)), Iter(std::end(a)), is_odd);
      assert(r == Iter(std::begin(a) + 1));
      assert(a[0] == P(1, 1));
      assert(a[1] == P(0, 2));
      assert(a[2] == P(2, 1));
      assert(a[3] == P(2, 2));
      assert(a[4] == P(4, 1));
      assert(a[5] == P(4, 2));
      assert(a[6] == P(6, 1));
      assert(a[7] == P(6, 2));
      assert(a[8] == P(8, 1));
      assert(a[9] == P(8, 2));
    }
    { // Check all false but last true
      P a[]  = {P(0, 1), P(0, 2), P(2, 1), P(2, 2), P(4, 1), P(4, 2), P(6, 1), P(6, 2), P(8, 1), P(1, 2)};
      auto r = std::stable_partition(policy, Iter(std::begin(a)), Iter(std::end(a)), is_odd);
      assert(r == Iter(std::begin(a) + 1));
      assert(a[0] == P(1, 2));
      assert(a[1] == P(0, 1));
      assert(a[2] == P(0, 2));
      assert(a[3] == P(2, 1));
      assert(a[4] == P(2, 2));
      assert(a[5] == P(4, 1));
      assert(a[6] == P(4, 2));
      assert(a[7] == P(6, 1));
      assert(a[8] == P(6, 2));
      assert(a[9] == P(8, 1));
    }
    { // Check all true but first false
      P a[]  = {P(0, 1), P(1, 2), P(3, 1), P(3, 2), P(5, 1), P(5, 2), P(7, 1), P(7, 2), P(9, 1), P(9, 2)};
      auto r = std::stable_partition(policy, Iter(std::begin(a)), Iter(std::end(a)), is_odd);
      assert(r == Iter(std::end(a) - 1));
      assert(a[0] == P(1, 2));
      assert(a[1] == P(3, 1));
      assert(a[2] == P(3, 2));
      assert(a[3] == P(5, 1));
      assert(a[4] == P(5, 2));
      assert(a[5] == P(7, 1));
      assert(a[6] == P(7, 2));
      assert(a[7] == P(9, 1));
      assert(a[8] == P(9, 2));
      assert(a[9] == P(0, 1));
    }
    { // Check all true but last false
      P a[]  = {P(1, 1), P(1, 2), P(3, 1), P(3, 2), P(5, 1), P(5, 2), P(7, 1), P(7, 2), P(9, 1), P(0, 2)};
      auto r = std::stable_partition(policy, Iter(std::begin(a)), Iter(std::end(a)), is_odd);
      assert(r == Iter(std::end(a) - 1));
      assert(a[0] == P(1, 1));
      assert(a[1] == P(1, 2));
      assert(a[2] == P(3, 1));
      assert(a[3] == P(3, 2));
      assert(a[4] == P(5, 1));
      assert(a[5] == P(5, 2));
      assert(a[6] == P(7, 1));
      assert(a[7] == P(7, 2));
      assert(a[8] == P(9, 1));
      assert(a[9] == P(0, 2));
    }
  }
};

int main(int, char**) {
  types::for_each(types::bidirectional_iterator_list<std::pair<int, int>*>{}, TestIteratorWithPolicies<Test>{});
  return 0;
}
