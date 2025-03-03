//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// UNSUPPORTED: c++03, c++11, c++14

// UNSUPPORTED: libcpp-has-no-incomplete-pstl

// template<class ExecutionPolicy, class ForwardIterator, class Size, class T>
//   ForwardIterator fill_n(ExecutionPolicy&& exec,
//                          ForwardIterator first, Size n, const T& value);

#include <algorithm>
#include <cassert>
#include <vector>

#include "test_macros.h"
#include "test_execution_policies.h"
#include "test_iterators.h"
#include "type_algorithms.h"

EXECUTION_POLICY_SFINAE_TEST(fill_n);

static_assert(sfinae_test_fill_n<int, int*, int*, bool (*)(int)>);
static_assert(!sfinae_test_fill_n<std::execution::parallel_policy, int*, int*, int>);

template <class Iter>
struct Test {
  template <class Policy>
  void operator()(Policy&& policy) {
    { // simple test
      int a[4];
      std::fill_n(policy, Iter(std::begin(a)), std::size(a), 33);
      assert(std::all_of(std::begin(a), std::end(a), [](int i) { return i == 33; }));
    }
    { // check that an empty range works
      int a[1] = {2};
      std::fill_n(policy, Iter(std::begin(a)), 0, 33);
      assert(a[0] == 2);
    }
    { // check that a one-element range works
      int a[1];
      std::fill_n(policy, Iter(std::begin(a)), std::size(a), 33);
      assert(std::all_of(std::begin(a), std::end(a), [](int i) { return i == 33; }));
    }
    { // check that a two-element range works
      int a[2];
      std::fill_n(policy, Iter(std::begin(a)), std::size(a), 33);
      assert(std::all_of(std::begin(a), std::end(a), [](int i) { return i == 33; }));
    }
    { // check that a large range works
      std::vector<int> a(234, 2);
      std::fill_n(policy, Iter(std::data(a)), std::size(a), 33);
      assert(std::all_of(std::begin(a), std::end(a), [](int i) { return i == 33; }));
    }
  }
};

int main(int, char**) {
  types::for_each(types::forward_iterator_list<int*>{}, TestIteratorWithPolicies<Test>{});

  return 0;
}
