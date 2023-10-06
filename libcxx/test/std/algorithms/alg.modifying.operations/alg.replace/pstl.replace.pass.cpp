//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: libcpp-has-no-incomplete-pstl

// template<class ExecutionPolicy, class ForwardIterator, class T>
//   void replace(ExecutionPolicy&& exec,
//                ForwardIterator first, ForwardIterator last,
//                const T& old_value, const T& new_value);

#include <algorithm>
#include <array>
#include <cassert>
#include <vector>

#include "type_algorithms.h"
#include "test_execution_policies.h"
#include "test_iterators.h"

template <class Iter>
struct Test {
  template <class ExecutionPolicy>
  void operator()(ExecutionPolicy&& policy) {
    { // simple test
      std::array a = {1, 2, 3, 4, 5, 6, 7, 8};
      std::replace(policy, Iter(std::begin(a)), Iter(std::end(a)), 3, 6);
      assert((a == std::array{1, 2, 6, 4, 5, 6, 7, 8}));
    }

    { // empty range works
      std::array<int, 0> a = {};
      std::replace(policy, Iter(std::begin(a)), Iter(std::end(a)), 3, 6);
    }

    { // non-empty range without a match works
      std::array a = {1, 2};
      std::replace(policy, Iter(std::begin(a)), Iter(std::end(a)), 3, 6);
    }

    { // single element range works
      std::array a = {3};
      std::replace(policy, Iter(std::begin(a)), Iter(std::end(a)), 3, 6);
      assert((a == std::array{6}));
    }

    { // two element range works
      std::array a = {3, 4};
      std::replace(policy, Iter(std::begin(a)), Iter(std::end(a)), 3, 6);
      assert((a == std::array{6, 4}));
    }

    { // multiple matching elements work
      std::array a = {1, 2, 3, 4, 3, 3, 5, 6, 3};
      std::replace(policy, Iter(std::begin(a)), Iter(std::end(a)), 3, 9);
      assert((a == std::array{1, 2, 9, 4, 9, 9, 5, 6, 9}));
    }

    { // large range works
      std::vector<int> a(150, 3);
      a[45] = 5;
      std::replace(policy, Iter(std::data(a)), Iter(std::data(a) + std::size(a)), 3, 6);

      std::vector<int> comp(150, 6);
      comp[45] = 5;
      assert(std::equal(a.begin(), a.end(), comp.begin()));
    }
  }
};

int main(int, char**) {
  types::for_each(types::forward_iterator_list<int*>{}, TestIteratorWithPolicies<Test>{});

  return 0;
}
