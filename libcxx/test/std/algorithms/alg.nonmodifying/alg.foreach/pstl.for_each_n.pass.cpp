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

// template<class ExecutionPolicy, class ForwardIterator, class Size, class Function>
//   ForwardIterator for_each_n(ExecutionPolicy&& exec, ForwardIterator first, Size n,
//                              Function f);

#include <algorithm>
#include <atomic>
#include <cassert>
#include <type_traits>
#include <vector>

#include "test_macros.h"
#include "test_execution_policies.h"
#include "test_iterators.h"
#include "type_algorithms.h"

EXECUTION_POLICY_SFINAE_TEST(for_each_n);

static_assert(sfinae_test_for_each_n<int, int*, int, bool (*)(int)>);
static_assert(!sfinae_test_for_each_n<std::execution::parallel_policy, int*, int, bool (*)(int)>);

template <class Iter>
struct Test {
  template <class Policy>
  void operator()(Policy&& policy) {
    { // Check the return type
      int a[]  = {0};
      auto res = std::for_each_n(policy, Iter(std::begin(a)), 0, [](int) {});
      static_assert(std::is_same_v<decltype(res), Iter>);
    }
    { // Check that every single element is visited
      int sizes[] = {0, 1, 2, 100};
      for (auto size : sizes) {
        std::vector<int> a(size);
        std::vector<Bool> called(size);
        auto res = std::for_each_n(policy, Iter(std::data(a)), std::size(a), [&](int& v) {
          assert(!called[&v - a.data()]);
          called[&v - a.data()] = true;
        });
        assert(std::all_of(std::begin(called), std::end(called), [](bool b) { return b; }));
        assert(res == Iter(std::data(a) + std::size(a)));
      }
    }
  }
};

int main(int, char**) {
  types::for_each(types::forward_iterator_list<int*>{}, TestIteratorWithPolicies<Test>{});

  return 0;
}
