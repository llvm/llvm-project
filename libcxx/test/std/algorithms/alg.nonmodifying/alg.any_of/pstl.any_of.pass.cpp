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

// template<class ExecutionPolicy, class ForwardIterator, class Predicate>
//   bool any_of(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last,
//               Predicate pred);

#include <algorithm>
#include <cassert>
#include <vector>

#include "test_macros.h"
#include "test_execution_policies.h"
#include "test_iterators.h"

EXECUTION_POLICY_SFINAE_TEST(any_of);

static_assert(sfinae_test_any_of<int, int*, int*, bool (*)(int)>);
static_assert(!sfinae_test_any_of<std::execution::parallel_policy, int*, int*, bool (*)(int)>);

template <class Iter>
struct Test {
  template <class Policy>
  void operator()(Policy&& policy) {
    int a[] = {1, 2, 3, 4, 5, 6, 7, 8};
    // simple test
    assert(std::any_of(policy, Iter(std::begin(a)), Iter(std::end(a)), [](int i) { return i < 9; }));
    assert(!std::any_of(policy, Iter(std::begin(a)), Iter(std::end(a)), [](int i) { return i > 8; }));

    // check that an empty range works
    assert(!std::any_of(policy, Iter(std::begin(a)), Iter(std::begin(a)), [](int) { return false; }));

    // check that false is returned if no element satisfies the condition
    assert(!std::any_of(policy, Iter(std::begin(a)), Iter(std::end(a)), [](int i) { return i == 9; }));

    // check that true is returned if only one elements satisfies the condition
    assert(std::any_of(policy, Iter(std::begin(a)), Iter(std::end(a)), [](int i) { return i == 1; }));

    // check that a one-element range works
    assert(std::any_of(policy, Iter(std::begin(a)), Iter(std::begin(a) + 1), [](int i) { return i == 1; }));

    // check that a two-element range works
    assert(std::any_of(policy, Iter(std::begin(a)), Iter(std::begin(a) + 2), [](int i) { return i == 2; }));

    // check that a large number of elements works
    std::vector<int> vec(100, 2);
    vec[96] = 3;
    assert(std::any_of(policy, Iter(vec.data()), Iter(vec.data() + vec.size()), [](int i) { return i == 3; }));
  }
};

int main(int, char**) {
  types::for_each(types::forward_iterator_list<int*>{}, TestIteratorWithPolicies<Test>{});

  return 0;
}
