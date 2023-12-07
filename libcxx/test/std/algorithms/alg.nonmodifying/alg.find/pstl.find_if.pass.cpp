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
//   ForwardIterator find_if(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last,
//                           Predicate pred);

#include <algorithm>
#include <cassert>
#include <vector>

#include "test_macros.h"
#include "test_execution_policies.h"
#include "test_iterators.h"

EXECUTION_POLICY_SFINAE_TEST(find_if);

static_assert(sfinae_test_find_if<int, int*, int*, bool (*)(int)>);
static_assert(!sfinae_test_find_if<std::execution::parallel_policy, int*, int*, int>);

template <class Iter>
struct Test {
  template <class Policy>
  void operator()(Policy&& policy) {
    int a[] = {1, 2, 3, 4, 5, 6, 7, 8};

    // simple test
    assert(base(std::find_if(policy, Iter(std::begin(a)), Iter(std::end(a)), [](int i) { return i == 3; })) == a + 2);

    // check that last is returned if no element matches
    assert(base(std::find_if(policy, Iter(std::begin(a)), Iter(std::end(a)), [](int i) { return i == 0; })) ==
           std::end(a));

    // check that the first element is returned
    assert(base(std::find_if(policy, Iter(std::begin(a)), Iter(std::end(a)), [](int i) { return i == 1; })) ==
           std::begin(a));

    // check that an empty range works
    assert(base(std::find_if(policy, Iter(std::begin(a)), Iter(std::begin(a)), [](int i) { return i == 1; })) ==
           std::begin(a));

    // check that a one-element range works
    assert(base(std::find_if(policy, Iter(std::begin(a)), Iter(std::begin(a) + 1), [](int i) { return i == 1; })) ==
           std::begin(a));

    // check that a two-element range works
    assert(base(std::find_if(policy, Iter(std::begin(a)), Iter(std::begin(a) + 2), [](int i) { return i == 2; })) ==
           std::begin(a) + 1);

    // check that a large number of elements works
    std::vector<int> vec(200, 4);
    vec[176] = 5;
    assert(base(std::find_if(policy, Iter(std::data(vec)), Iter(std::data(vec) + std::size(vec)), [](int i) {
             return i == 5;
           })) == std::data(vec) + 176);
  }
};

int main(int, char**) {
  types::for_each(types::forward_iterator_list<int*>{}, TestIteratorWithPolicies<Test>{});

#ifndef TEST_HAS_NO_EXCEPTIONS
  std::set_terminate(terminate_successful);
  int a[] = {1, 2};
  try {
    (void)std::find_if(std::execution::par, std::begin(a), std::end(a), [](int) -> bool { throw int{}; });
  } catch (int) {
    assert(false);
  }
#endif

  return 0;
}
