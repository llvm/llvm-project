//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// UNSUPPORTED: libcpp-has-no-incomplete-pstl

// <algorithm>

// template<class ExecutionPolicy, class ForwardIterator, class T>
//   typename iterator_traits<ForwardIterator>::difference_type
//     count(ExecutionPolicy&& exec,
//           ForwardIterator first, ForwardIterator last, const T& value);

#include <algorithm>
#include <array>
#include <cassert>
#include <vector>

#include "test_macros.h"
#include "test_execution_policies.h"
#include "test_iterators.h"

EXECUTION_POLICY_SFINAE_TEST(count);

static_assert(sfinae_test_count<int, int*, int*, bool (*)(int)>);
static_assert(!sfinae_test_count<std::execution::parallel_policy, int*, int*, int>);

template <class Iter>
struct Test {
  template <class Policy>
  void operator()(Policy&& policy) {
    { // simple test
      int a[]            = {1, 2, 3, 4, 5};
      decltype(auto) ret = std::count(policy, std::begin(a), std::end(a), 3);
      static_assert(std::is_same_v<decltype(ret), typename std::iterator_traits<Iter>::difference_type>);
      assert(ret == 1);
    }

    { // test that an empty range works
      std::array<int, 0> a;
      decltype(auto) ret = std::count(policy, std::begin(a), std::end(a), 3);
      static_assert(std::is_same_v<decltype(ret), typename std::iterator_traits<Iter>::difference_type>);
      assert(ret == 0);
    }

    { // test that a single-element range works
      int a[] = {1};
      decltype(auto) ret = std::count(policy, std::begin(a), std::end(a), 1);
      static_assert(std::is_same_v<decltype(ret), typename std::iterator_traits<Iter>::difference_type>);
      assert(ret == 1);
    }

    { // test that a two-element range works
      int a[] = {1, 3};
      decltype(auto) ret = std::count(policy, std::begin(a), std::end(a), 3);
      static_assert(std::is_same_v<decltype(ret), typename std::iterator_traits<Iter>::difference_type>);
      assert(ret == 1);
    }

    { // test that a three-element range works
      int a[] = {3, 1, 3};
      decltype(auto) ret = std::count(policy, std::begin(a), std::end(a), 3);
      static_assert(std::is_same_v<decltype(ret), typename std::iterator_traits<Iter>::difference_type>);
      assert(ret == 2);
    }

    { // test that a large range works
      std::vector<int> a(100, 2);
      decltype(auto) ret = std::count(policy, std::begin(a), std::end(a), 2);
      static_assert(std::is_same_v<decltype(ret), typename std::iterator_traits<Iter>::difference_type>);
      assert(ret == 100);
    }
  }
};

int main(int, char**) {
  types::for_each(types::forward_iterator_list<int*>{}, TestIteratorWithPolicies<Test>{});

  return 0;
}
