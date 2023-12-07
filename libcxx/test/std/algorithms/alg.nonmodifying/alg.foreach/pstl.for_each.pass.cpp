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

// template<class ExecutionPolicy, class ForwardIterator, class Function>
//   void for_each(ExecutionPolicy&& exec,
//                 ForwardIterator first, ForwardIterator last,
//                 Function f);

#include <algorithm>
#include <atomic>
#include <cassert>
#include <vector>

#include "test_macros.h"
#include "test_execution_policies.h"
#include "test_iterators.h"

EXECUTION_POLICY_SFINAE_TEST(for_each);

static_assert(sfinae_test_for_each<int, int*, int*, bool (*)(int)>);
static_assert(!sfinae_test_for_each<std::execution::parallel_policy, int*, int*, bool (*)(int)>);

template <class Iter>
struct Test {
  template <class Policy>
  void operator()(Policy&& policy) {
    int sizes[] = {0, 1, 2, 100};
    for (auto size : sizes) {
      std::vector<int> a(size);
      std::vector<Bool> called(size);
      std::for_each(policy, Iter(std::data(a)), Iter(std::data(a) + std::size(a)), [&](int& v) {
        assert(!called[&v - a.data()]);
        called[&v - a.data()] = true;
      });
      assert(std::all_of(std::begin(called), std::end(called), [](bool b) { return b; }));
    }
  }
};

int main(int, char**) {
  types::for_each(types::forward_iterator_list<int*>{}, TestIteratorWithPolicies<Test>{});

#ifndef TEST_HAS_NO_EXCEPTIONS
  std::set_terminate(terminate_successful);
  int a[] = {1, 2};
  try {
    std::for_each(std::execution::par, std::begin(a), std::end(a), [](int) { throw int{}; });
  } catch (int) {
    assert(false);
  }
#endif

  return 0;
}
