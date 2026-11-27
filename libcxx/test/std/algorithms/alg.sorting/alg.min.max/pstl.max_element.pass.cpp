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
//   ForwardIterator
//   max_element(ExecutionPolicy&& exec,
//               ForwardIterator first, ForwardIterator last);

#include <cstddef>
#include <algorithm>
#include <cassert>
#include <iterator>
#include <numeric>
#include <random>
#include <vector>

#include "test_execution_policies.h"
#include "test_iterators.h"
#include "test_macros.h"
#include "type_algorithms.h"
#include "runway_sample.h"

EXECUTION_POLICY_SFINAE_TEST(max_element);

static_assert(sfinae_test_max_element<int, int*, int*>);
static_assert(!sfinae_test_max_element<std::execution::parallel_policy, int*, int*>);

template <class Iter>
struct Test {
  template <class ExecutionPolicy>
  void operator()(ExecutionPolicy&& policy) {
    { // Check the return type
      int a[]  = {0};
      auto res = std::max_element(policy, Iter(std::begin(a)), Iter(std::end(a)));
      static_assert(std::is_same_v<decltype(res), Iter>);
    }
    { // Empty
      int a[] = {0};
      assert(std::max_element(policy, Iter(std::begin(a)), Iter(std::begin(a))) == Iter(std::begin(a)));
    }
    { // Single
      int a[] = {0};
      assert(std::max_element(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a)));
    }
    { // Two elements, second max
      int a[] = {0, 1};
      assert(std::max_element(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 1));
    }
    { // Two elements, first max
      int a[] = {1, 0};
      assert(std::max_element(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a)));
    }
    { // Three elements, first max
      int a[] = {2, 1, 0};
      assert(std::max_element(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a)));
    }
    { // Three elements, middle max
      int a[] = {0, 2, 1};
      assert(std::max_element(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 1));
    }
    { // Three elements, last max
      int a[] = {1, 0, 2};
      assert(std::max_element(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 2));
    }
    { // Four elements, first max
      int a[] = {3, 0, 2, 1};
      assert(std::max_element(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a)));
    }
    { // Four elements, all equal
      int a[] = {1, 1, 1, 1};
      assert(std::max_element(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a)));
    }
    { // Four elements, middle max
      int a[] = {0, 3, 2, 1};
      assert(std::max_element(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 1));
    }
    { // Four elements, middle max
      int a[] = {0, 2, 3, 1};
      assert(std::max_element(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 2));
    }
    { // Four elements, middle max and equal to consequent elements
      int a[] = {0, 3, 3, 3};
      assert(std::max_element(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 1));
    }
    { // Four elements, last max
      int a[] = {0, 1, 2, 3};
      assert(std::max_element(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 3));
    }
    { // Check for maximum among iotaed and shuffled elements
      std::mt19937 randomness;
      auto verify_max = [&](Iter first, Iter last) {
        Iter i = std::max_element(policy, first, last);
        if (first != last) {
          for (Iter j = first; j != last; ++j)
            assert(!(*j > *i));
        } else {
          assert(i == last);
        }
      };
      auto test_n_elem = [&](int N) {
        std::vector<int> a(N);
        std::iota(a.begin(), a.end(), 0);
        std::shuffle(a.begin(), a.end(), randomness);
        verify_max(Iter(a.data()), Iter(a.data() + N));
      };
      test_n_elem(3);
      test_n_elem(4);
      test_n_elem(5);
      test_n_elem(10);
      test_n_elem(100);
      test_n_elem(1000);
    }
    { // Check that the first one among equals is returned
      int a[1073];
      // Fill with 0 0 0 0 .... 2 1 2 1 ...
      for (std::size_t i = 0; i < std::size(a); ++i)
        a[i] = (i < 473) ? 0 : (i % 2 ? 2 : 1);
      assert(std::max_element(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 473));
    }
    { // Check that the maximum at sampled locations is found
      int a[1073];
      std::fill(std::begin(a), std::end(a), 0);
      runway_sample(std::size(a), [&](size_t i) {
        a[i] = 1;
        assert(std::max_element(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + i));
        a[i] = 0;
      });
    }
  }
};

int main(int, char**) {
  types::for_each(types::forward_iterator_list<const int*>{}, TestIteratorWithPolicies<Test>{});
  return 0;
}
