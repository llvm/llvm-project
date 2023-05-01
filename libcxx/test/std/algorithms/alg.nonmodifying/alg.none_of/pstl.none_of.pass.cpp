//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: with-pstl

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

EXECUTION_POLICY_SFINAE_TEST(none_of);

static_assert(sfinae_test_none_of<int, int*, int*, bool (*)(int)>);
static_assert(!sfinae_test_none_of<std::execution::parallel_policy, int*, int*, bool (*)(int)>);

template <class Iter>
struct Test {
  template <class Policy>
  void operator()(Policy&& policy) {
    int a[] = {1, 2, 3, 4, 5, 6, 7, 8};
    // simple test
    assert(std::none_of(policy, Iter(std::begin(a)), Iter(std::end(a)), [](int i) { return i > 9; }));
    assert(!std::none_of(policy, Iter(std::begin(a)), Iter(std::end(a)), [](int i) { return i >= 8; }));

    // check that an empty range works
    assert(std::none_of(policy, Iter(std::begin(a)), Iter(std::begin(a)), [](int) { return false; }));

    // check that true is returned if no element satisfies the condition
    assert(std::none_of(policy, Iter(std::begin(a)), Iter(std::end(a)), [](int i) { return i == 9; }));

    // check that false is returned if only one elements satisfies the condition
    assert(!std::none_of(policy, Iter(std::begin(a)), Iter(std::end(a)), [](int i) { return i == 1; }));

    // check that a one-element range works
    assert(std::none_of(policy, Iter(std::begin(a)), Iter(std::begin(a) + 1), [](int i) { return i != 1; }));

    // check that a two-element range works
    assert(std::none_of(policy, Iter(std::begin(a)), Iter(std::begin(a) + 2), [](int i) { return i > 2; }));

    // check that a large number of elements works
    std::vector<int> vec(100);
    std::fill(vec.begin(), vec.end(), 3);
    assert(std::none_of(Iter(vec.data()), Iter(vec.data() + vec.size()), [](int i) { return i != 3; }));
  }
};

int main(int, char**) {
  types::for_each(types::forward_iterator_list<int*>{}, TestIteratorWithPolicies<Test>{});

#ifndef TEST_HAS_NO_EXCEPTIONS
  std::set_terminate(terminate_successful);
  int a[] = {1, 2};
  try {
    (void)std::none_of(std::execution::par, std::begin(a), std::end(a), [](int i) -> bool { throw i; });
  } catch (int) {
    assert(false);
  }
#endif

  return 0;
}
