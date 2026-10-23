//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++17

// UNSUPPORTED: libcpp-has-no-incomplete-pstl

// <algorithm>

// template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2, class Comp>
//   bool
//   includes(ExecutionPolicy&& exec,
//            ForwardIterator1 first1,
//            ForwardIterator1 last1,
//            ForwardIterator2 first2,
//            ForwardIterator2 last2,
//            Comp compare);

#include <cstddef>
#include <algorithm>
#include <cassert>
#include <functional>
#include <iterator>
#include <numeric>
#include <random>
#include <vector>

#include "test_execution_policies.h"
#include "test_iterators.h"
#include "test_macros.h"
#include "type_algorithms.h"

EXECUTION_POLICY_SFINAE_TEST(includes);

static_assert(sfinae_test_includes<int, int*, int*, int*, int*, bool (*)(int, int)>);
static_assert(!sfinae_test_includes<std::execution::parallel_policy, int*, int*, int*, int*, bool (*)(int, int)>);

template <class Iter1, class Iter2>
struct Test {
  template <class ExecutionPolicy>
  void operator()(ExecutionPolicy&& policy) {
    { // Check the return type
      int a[]  = {1};
      auto res = std::includes(
          policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(a)), Iter2(std::end(a)), std::greater{});
      static_assert(std::is_same_v<decltype(res), bool>);
    }
    { // Both ranges are empty
      int a[] = {1};
      int b[] = {1};
      assert(std::includes(
          policy,
          Iter1(std::begin(a)),
          Iter1(std::begin(a)),
          Iter2(std::begin(b)),
          Iter2(std::begin(b)),
          std::greater{}));
    }
    { // Empty haystack, non-empty needly
      int a[] = {1};
      int b[] = {1};
      assert(!std::includes(
          policy,
          Iter1(std::begin(a)),
          Iter1(std::begin(a)),
          Iter2(std::begin(b)),
          Iter2(std::end(b)),
          std::greater{}));
    }
    { // Non-empty haystack, empty needle
      int a[] = {1};
      int b[] = {1};
      assert(std::includes(
          policy,
          Iter1(std::begin(a)),
          Iter1(std::end(a)),
          Iter2(std::begin(b)),
          Iter2(std::begin(b)),
          std::greater{}));
    }
    { // Haystack and needle are the same
      int a[] = {4, 4, 4, 4, 3, 3, 3, 2, 2, 1};
      assert(std::includes(
          policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(a)), Iter2(std::end(a)), std::greater{}));
    }
    { // Needle is a sparse subset
      int a[] = {4, 4, 4, 4, 3, 3, 3, 2, 2, 1};
      int b[] = {4, 2};
      assert(std::includes(
          policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(b)), Iter2(std::end(b)), std::greater{}));
    }
    { // Superset is not included in its subset
      int a[] = {4, 2};
      int b[] = {4, 4, 4, 4, 3, 3, 3, 2, 2, 1};
      assert(!std::includes(
          policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(b)), Iter2(std::end(b)), std::greater{}));
    }
    { // Haystack and needle are equal
      int a[] = {2, 1};
      int b[] = {2, 1};
      assert(std::includes(
          policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(b)), Iter2(std::end(b)), std::greater{}));
    }
    { // Partial overlap
      int a[] = {2, 1};
      int b[] = {4, 2};
      assert(!std::includes(
          policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(b)), Iter2(std::end(b)), std::greater{}));
    }
    { // Needle is one element present multiple times
      int a[] = {4, 4, 4, 4, 3, 3, 3, 2, 2, 1};
      int b[] = {3};
      assert(std::includes(
          policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(b)), Iter2(std::end(b)), std::greater{}));
    }
    { // Needle is two-element present multiple times
      int a[] = {4, 4, 4, 4, 3, 3, 3, 2, 2, 1};
      int b[] = {3, 3};
      assert(std::includes(
          policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(b)), Iter2(std::end(b)), std::greater{}));
    }
    { // Needle is three-element duplicate present once
      int a[] = {4, 4, 4, 4, 3, 3, 3, 2, 2, 1};
      int b[] = {3, 3, 3};
      assert(std::includes(
          policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(b)), Iter2(std::end(b)), std::greater{}));
    }
    { // Needle is three-element duplicate not present
      int a[] = {4, 4, 4, 4, 3, 3, 3, 2, 2, 1};
      int b[] = {3, 3, 3, 3};
      assert(!std::includes(
          policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(b)), Iter2(std::end(b)), std::greater{}));
    }
  }
};

int main(int, char**) {
  types::for_each(types::forward_iterator_list<const int*>{}, types::apply_type_identity{[](auto v) {
                    using Iter = typename decltype(v)::type;
                    types::for_each(
                        types::forward_iterator_list<const int*>{},
                        TestIteratorWithPolicies<types::partial_instantiation<Test, Iter>::template apply>{});
                  }});
  return 0;
}
