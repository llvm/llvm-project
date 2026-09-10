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

// template<class ExecutionPolicy, class ForwardIterator, class Compare>
//   pair<ForwardIterator, ForwardIterator>
//   minmax_element(ExecutionPolicy&& exec,
//                  ForwardIterator first, ForwardIterator last,
//                  Compare comp);

#include <cstddef>
#include <algorithm>
#include <cassert>
#include <functional>
#include <iterator>
#include <numeric>
#include <random>
#include <vector>

#include "test_execution_policies.h"
#include "test_iterators.h"
#include "test_macros.h"
#include "type_algorithms.h"
#include "runway_sample.h"

EXECUTION_POLICY_SFINAE_TEST(minmax_element);

static_assert(sfinae_test_minmax_element<int, int*, int*, bool (*)(int, int)>);
static_assert(!sfinae_test_minmax_element<std::execution::parallel_policy, int*, int*, bool (*)(int, int)>);

template <class Iter>
struct Test {
  template <class ExecutionPolicy>
  void operator()(ExecutionPolicy&& policy) {
    using IterPair = std::pair<Iter, Iter>;
    std::greater<int> comp;
    { // Check the return type
      int a[]  = {0};
      auto res = std::minmax_element(policy, Iter(std::begin(a)), Iter(std::end(a)), comp);
      static_assert(std::is_same_v<decltype(res), IterPair>);
    }
    { // Empty
      int a[] = {0};
      assert(std::minmax_element(policy, Iter(std::begin(a)), Iter(std::begin(a)), comp) ==
             IterPair(std::begin(a), std::begin(a)));
    }
    { // Single
      int a[] = {0};
      assert(std::minmax_element(policy, Iter(std::begin(a)), Iter(std::end(a)), comp) ==
             IterPair(std::begin(a), std::begin(a)));
    }
    { // Two elements, second min, first max
      int a[] = {0, 1};
      assert(std::minmax_element(policy, Iter(std::begin(a)), Iter(std::end(a)), comp) ==
             IterPair(std::begin(a) + 1, std::begin(a)));
    }
    { // Two elements, first min, second max
      int a[] = {1, 0};
      assert(std::minmax_element(policy, Iter(std::begin(a)), Iter(std::end(a)), comp) ==
             IterPair(std::begin(a), std::begin(a) + 1));
    }
    { // Three elements, third min, first max
      int a[] = {0, 1, 2};
      assert(std::minmax_element(policy, Iter(std::begin(a)), Iter(std::end(a)), comp) ==
             IterPair(std::begin(a) + 2, std::begin(a)));
    }
    { // Three elements, third min, middle max
      int a[] = {1, 0, 2};
      assert(std::minmax_element(policy, Iter(std::begin(a)), Iter(std::end(a)), comp) ==
             IterPair(std::begin(a) + 2, std::begin(a) + 1));
    }
    { // Three elements, first min, third max
      int a[] = {1, 0, 0};
      assert(std::minmax_element(policy, Iter(std::begin(a)), Iter(std::end(a)), comp) ==
             IterPair(std::begin(a), std::begin(a) + 2));
    }
    { // Three elements, first min, third max
      int a[] = {1, 1, 0};
      assert(std::minmax_element(policy, Iter(std::begin(a)), Iter(std::end(a)), comp) ==
             IterPair(std::begin(a), std::begin(a) + 2));
    }
    { // Three elements, first min, third max
      int a[] = {2, 1, 0};
      assert(std::minmax_element(policy, Iter(std::begin(a)), Iter(std::end(a)), comp) ==
             IterPair(std::begin(a), std::begin(a) + 2));
    }
    { // Four elements, fourth min, first max
      int a[] = {0, 1, 2, 3};
      assert(std::minmax_element(policy, Iter(std::begin(a)), Iter(std::end(a)), comp) ==
             IterPair(std::begin(a) + 3, std::begin(a)));
    }
    { // Four elements, all equal
      int a[] = {1, 1, 1, 1};
      assert(std::minmax_element(policy, Iter(std::begin(a)), Iter(std::end(a)), comp) ==
             IterPair(std::begin(a), std::begin(a) + 3));
    }
    { // Four elements, first min, middle max
      int a[] = {3, 0, 2, 1};
      assert(std::minmax_element(policy, Iter(std::begin(a)), Iter(std::end(a)), comp) ==
             IterPair(std::begin(a), std::begin(a) + 1));
    }
    { // Four elements, first min, fourth max
      int a[] = {3, 0, 0, 0};
      assert(std::minmax_element(policy, Iter(std::begin(a)), Iter(std::end(a)), comp) ==
             IterPair(std::begin(a), std::begin(a) + 3));
    }
    { // Four elements, first min, fourth max
      int a[] = {3, 3, 0, 0};
      assert(std::minmax_element(policy, Iter(std::begin(a)), Iter(std::end(a)), comp) ==
             IterPair(std::begin(a), std::begin(a) + 3));
    }
    { // Four elements, first min, fourth max
      int a[] = {3, 2, 1, 0};
      assert(std::minmax_element(policy, Iter(std::begin(a)), Iter(std::end(a)), comp) ==
             IterPair(std::begin(a), std::begin(a) + 3));
    }
    { // Five elements, all equal
      int a[] = {1, 1, 1, 1, 1};
      assert(std::minmax_element(policy, Iter(std::begin(a)), Iter(std::end(a)), comp) ==
             IterPair(std::begin(a), std::begin(a) + 4));
    }
    { // Five elements, ascending
      int a[] = {1, 2, 3, 4, 5};
      assert(std::minmax_element(policy, Iter(std::begin(a)), Iter(std::end(a)), comp) ==
             IterPair(std::begin(a) + 4, std::begin(a)));
    }
    { // Five elements, descending
      int a[] = {5, 4, 3, 2, 1};
      assert(std::minmax_element(policy, Iter(std::begin(a)), Iter(std::end(a)), comp) ==
             IterPair(std::begin(a), std::begin(a) + 4));
    }
    { // Five elements, 2 flat regions, descending
      int a[] = {1, 1, 1, 0, 0};
      assert(std::minmax_element(policy, Iter(std::begin(a)), Iter(std::end(a)), comp) ==
             IterPair(std::begin(a), std::begin(a) + 4));
    }
    { // Five elements, 2 flat regions, ascending
      int a[] = {0, 0, 1, 1, 1};
      assert(std::minmax_element(policy, Iter(std::begin(a)), Iter(std::end(a)), comp) ==
             IterPair(std::begin(a) + 2, std::begin(a) + 1));
    }
    { // Five elements, alternating 1/0 pattern
      int a[] = {1, 0, 1, 0, 1};
      assert(std::minmax_element(policy, Iter(std::begin(a)), Iter(std::end(a)), comp) ==
             IterPair(std::begin(a), std::begin(a) + 3));
    }
    { // Check for minimum and maximum among iotaed and shuffled elements
      std::mt19937 randomness;
      auto verify_min = [&](Iter first, Iter last) {
        IterPair i = std::minmax_element(policy, first, last, comp);
        if (first != last) {
          for (Iter j = first; j != last; ++j) {
            assert(!(*i.first < *j));
            assert(!(*j < *i.second));
          }
        } else {
          assert(i == IterPair(last, last));
        }
      };
      auto test_n_elem = [&](int N) {
        std::vector<int> a(N);
        std::iota(a.begin(), a.end(), 0);
        std::shuffle(a.begin(), a.end(), randomness);
        verify_min(Iter(a.data()), Iter(a.data() + N));
      };
      test_n_elem(3);
      test_n_elem(4);
      test_n_elem(5);
      test_n_elem(10);
      test_n_elem(100);
      test_n_elem(1000);
    }
    { // Check that the first greatest is returned as the minimum and the last smallest is returned as the maximum
      int a[1073];
      // Fill with 2 2 2 2 .... 0 1 0 1 ...
      for (std::size_t i = 0; i < std::size(a); ++i)
        a[i] = (i < 473) ? 2 : (i % 2 ? 0 : 1);
      assert(std::minmax_element(policy, Iter(std::begin(a)), Iter(std::end(a)), comp) ==
             IterPair(std::begin(a), std::begin(a) + 1071));
    }
    { // Check that the minimum/maximum at sampled locations is found
      int a[1073];
      std::fill(std::begin(a), std::end(a), 1);
      runway_sample(std::size(a) - 1, [&](size_t i) {
        if (i == 0)
          return;
        a[i] = 0;
        assert(std::minmax_element(policy, Iter(std::begin(a)), Iter(std::end(a)), comp) ==
               IterPair(std::begin(a), std::begin(a) + i));
        a[i] = 2;
        assert(std::minmax_element(policy, Iter(std::begin(a)), Iter(std::end(a)), comp) ==
               IterPair(std::begin(a) + i, std::begin(a) + std::size(a) - 1));
        a[i] = 1;
      });
    }
  }
};

int main(int, char**) {
  types::for_each(types::forward_iterator_list<const int*>{}, TestIteratorWithPolicies<Test>{});
  return 0;
}
