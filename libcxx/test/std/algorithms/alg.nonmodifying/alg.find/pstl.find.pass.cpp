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
//   ForwardIterator find(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last,
//                        const T& value);

#include <algorithm>
#include <cassert>
#include <vector>

#include "test_macros.h"
#include "test_execution_policies.h"
#include "test_iterators.h"

EXECUTION_POLICY_SFINAE_TEST(find);

static_assert(sfinae_test_find<int, int*, int*, bool (*)(int)>);
static_assert(!sfinae_test_find<std::execution::parallel_policy, int*, int*, int>);

template <class Iter>
struct Test {
  template <class Policy>
  void operator()(Policy&& policy) {
    int a[] = {1, 2, 3, 4, 5, 6, 7, 8};

    // simple test
    assert(base(std::find(policy, Iter(std::begin(a)), Iter(std::end(a)), 3)) == a + 2);

    // check that last is returned if no element matches
    assert(base(std::find(policy, Iter(std::begin(a)), Iter(std::end(a)), 0)) == std::end(a));

    // check that the first element is returned
    assert(base(std::find(policy, Iter(std::begin(a)), Iter(std::end(a)), 1)) == std::begin(a));

    // check that an empty range works
    assert(base(std::find(policy, Iter(std::begin(a)), Iter(std::begin(a)), 1)) == std::begin(a));

    // check that a one-element range works
    assert(base(std::find(policy, Iter(std::begin(a)), Iter(std::begin(a) + 1), 1)) == std::begin(a));

    // check that a two-element range works
    assert(base(std::find(policy, Iter(std::begin(a)), Iter(std::begin(a) + 2), 2)) == std::begin(a) + 1);

    // check that a large number of elements works
    std::vector<int> vec(200, 4);
    vec[176] = 5;
    assert(base(std::find(policy, Iter(std::data(vec)), Iter(std::data(vec) + std::size(vec)), 5)) ==
           std::data(vec) + 176);
  }
};

struct ThrowOnCompare {};

#ifndef TEST_HAS_NO_EXCEPTIONS
bool operator==(ThrowOnCompare, ThrowOnCompare) { throw int{}; }
#endif

int main(int, char**) {
  types::for_each(types::forward_iterator_list<int*>{}, TestIteratorWithPolicies<Test>{});

#ifndef TEST_HAS_NO_EXCEPTIONS
  std::set_terminate(terminate_successful);
  ThrowOnCompare a[2];
  try {
    (void)std::find(std::execution::par, std::begin(a), std::end(a), ThrowOnCompare{});
  } catch (int) {
    assert(false);
  }
#endif

  return 0;
}
