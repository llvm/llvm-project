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

// template<class ExecutionPolicy, class BidirectionalIterator>
//   void
//   inplace_merge(ExecutionPolicy&& exec,
//                 BidirectionalIterator first,
//                 BidirectionalIterator middle,
//                 BidirectionalIterator last);

#include <cstddef>
#include <algorithm>
#include <cassert>
#include <iterator>
#include <numeric>
#include <string>
#include <vector>

#include "constexpr_random.h"
#include "test_execution_policies.h"
#include "test_iterators.h"
#include "test_macros.h"
#include "type_algorithms.h"

EXECUTION_POLICY_SFINAE_TEST(inplace_merge);

static_assert(sfinae_test_inplace_merge<int, int*, int*, int*>);
static_assert(!sfinae_test_inplace_merge<std::execution::parallel_policy, int*, int*, int*>);

template <class Iter>
struct TestInt {
  template <class ExecutionPolicy>
  void operator()(ExecutionPolicy&& policy) {
    { // Check the return type
      int a[] = {0};
      static_assert(std::is_same_v<
                    decltype(std::inplace_merge(policy, Iter(std::begin(a)), Iter(std::begin(a)), Iter(std::begin(a)))),
                    void>);
    }
    { // Check empty
      int a[] = {0};
      std::inplace_merge(policy, Iter(std::begin(a)), Iter(std::begin(a)), Iter(std::begin(a)));
      assert(a[0] == 0);
    }
    { // Check single
      int a[] = {0};
      std::inplace_merge(policy, Iter(std::begin(a)), Iter(std::begin(a)), Iter(std::end(a)));
      assert(a[0] == 0);
      std::inplace_merge(policy, Iter(std::begin(a)), Iter(std::end(a)), Iter(std::end(a)));
      assert(a[0] == 0);
    }
    { // Check two
      int a[] = {1, 0};
      std::inplace_merge(policy, Iter(std::begin(a)), Iter(std::begin(a)), Iter(std::end(a)));
      assert(a[0] == 1);
      assert(a[1] == 0);
      std::inplace_merge(policy, Iter(std::begin(a)), Iter(std::end(a)), Iter(std::end(a)));
      assert(a[0] == 1);
      assert(a[1] == 0);
      std::inplace_merge(policy, Iter(std::begin(a)), Iter(std::begin(a) + 1), Iter(std::end(a)));
      assert(a[0] == 0);
      assert(a[1] == 1);
    }

    support::simple_random_generator randomness;
    auto test_shuffled_iota = [&](std::size_t count, std::size_t middle) {
      assert(middle <= count);
      std::vector<int> a(count);
      std::iota(a.begin(), a.end(), 0);
      support::shuffle(a.begin(), a.end(), randomness);
      std::sort(a.begin(), a.begin() + middle);
      std::sort(a.begin() + middle, a.end());
      std::inplace_merge(policy, Iter(a.data()), Iter(a.data() + middle), Iter(a.data() + count));
      if (count > 0) {
        assert(a[0] == 0);
        assert(a[count - 1] == static_cast<int>(count - 1));
        assert(std::is_sorted(a.begin(), a.end()));
      }
    };

    { // Check three
      test_shuffled_iota(3, 0);
      test_shuffled_iota(3, 1);
      test_shuffled_iota(3, 2);
      test_shuffled_iota(3, 3);
    }
    { // Check four
      test_shuffled_iota(4, 0);
      test_shuffled_iota(4, 1);
      test_shuffled_iota(4, 2);
      test_shuffled_iota(4, 3);
      test_shuffled_iota(4, 4);
    }
    { // Check 50
      test_shuffled_iota(50, 0);
      test_shuffled_iota(50, 15);
      test_shuffled_iota(50, 25);
      test_shuffled_iota(50, 35);
      test_shuffled_iota(50, 50);
    }
    { // Check 100
      test_shuffled_iota(100, 0);
      test_shuffled_iota(100, 25);
      test_shuffled_iota(100, 50);
      test_shuffled_iota(100, 75);
      test_shuffled_iota(100, 100);
    }
    { // Check 1000
      test_shuffled_iota(1000, 0);
      test_shuffled_iota(1000, 250);
      test_shuffled_iota(1000, 500);
      test_shuffled_iota(1000, 750);
      test_shuffled_iota(1000, 1000);
    }
  }
};

template <class Iter>
struct TestString {
  template <class ExecutionPolicy>
  void operator()(ExecutionPolicy&& policy) {
    support::simple_random_generator randomness;
    for (std::size_t count : {0, 1, 2, 3, 10, 100}) {
      std::vector<std::string> expected(count);
      for (std::size_t i = 0; i < count; ++i) {
        std::string num = std::to_string(i);
        expected[i]     = std::string(32 - num.size(), '0') + num;
      }
      std::size_t middle         = count / 2;
      std::vector<std::string> a = expected;
      support::shuffle(a.begin(), a.end(), randomness);
      std::sort(a.begin(), a.begin() + middle);
      std::sort(a.begin() + middle, a.end());
      std::inplace_merge(policy, Iter(a.data()), Iter(a.data() + middle), Iter(a.data() + count));
      assert(a == expected);
    }
  }
};

int main(int, char**) {
  types::for_each(types::bidirectional_iterator_list<int*>{}, TestIteratorWithPolicies<TestInt>{});
  types::for_each(types::bidirectional_iterator_list<std::string*>{}, TestIteratorWithPolicies<TestString>{});
  return 0;
}
