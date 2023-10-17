//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// UNSUPPORTED: libcpp-has-no-incomplete-pstl

// template<class ExecutionPolicy, class ForwardIterator, class Size, class Generator>
//   ForwardIterator generate_n(ExecutionPolicy&& exec,
//                              ForwardIterator first, Size n, Generator gen);

#include <algorithm>
#include <cassert>
#include <vector>

#include "test_iterators.h"
#include "test_execution_policies.h"
#include "type_algorithms.h"

template <class Iter>
struct Test {
  template <class ExecutionPolicy>
  void operator()(ExecutionPolicy&& policy) {
    { // simple test
      int a[10];
      std::generate_n(policy, Iter(std::begin(a)), std::size(a), []() { return 1; });
      assert(std::all_of(std::begin(a), std::end(a), [](int i) { return i == 1; }));
    }
    { // empty range works
      int a[10] {3};
      std::generate_n(policy, Iter(std::begin(a)), 0, []() { return 1; });
      assert(a[0] == 3);
    }
    { // single-element range works
      int a[] {3};
      std::generate_n(policy, Iter(std::begin(a)), std::size(a), []() { return 5; });
      assert(a[0] == 5);
    }
    { // large range works
      std::vector<int> vec(150, 4);
      std::generate_n(policy, Iter(std::data(vec)), std::size(vec), []() { return 5; });
      assert(std::all_of(std::begin(vec), std::end(vec), [](int i) { return i == 5; }));
    }
  }
};

int main(int, char**) {
  types::for_each(types::forward_iterator_list<int*>{}, TestIteratorWithPolicies<Test>{});

  return 0;
}
