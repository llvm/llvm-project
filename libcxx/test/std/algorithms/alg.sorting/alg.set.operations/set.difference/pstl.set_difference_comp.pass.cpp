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

// template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2,
//          class ForwardOutIterator, class Compare>
//   ForwardOutIterator
//   set_difference(ExecutionPolicy&& exec,
//                  ForwardIterator1 first1,
//                  ForwardIterator1 last1,
//                  ForwardIterator2 first2,
//                  ForwardIterator2 last2,
//                  ForwardOutIterator result,
//                  Compare comp);

#include <cstddef>
#include <algorithm>
#include <cassert>
#include <iterator>
#include <numeric>
#include <random>
#include <vector>

#include "test_execution_policies.h"
#include "test_iterators.h"
#include "test_macros.h"
#include "type_algorithms.h"
#include "../../sortable_helpers.h"

EXECUTION_POLICY_SFINAE_TEST(set_difference);

static_assert(sfinae_test_set_difference<int, int*, int*, int*, int*, int*, bool (*)(int, int)>);
static_assert(
    !sfinae_test_set_difference<std::execution::parallel_policy, int*, int*, int*, int*, int*, bool (*)(int, int)>);

template <class Iter1, class Iter2>
struct Test {
  template <class ExecutionPolicy>
  void operator()(ExecutionPolicy&& policy) {
    {
      using T   = TrivialSortableWithComp;
      auto comp = T::Comparator{};
      { // Check the result type
        T a[]    = {0};
        T b[]    = {0};
        T c[2]   = {};
        auto res = std::set_difference(
            policy,
            Iter1(std::begin(a)),
            Iter1(std::end(a)),
            Iter2(std::begin(b)),
            Iter2(std::end(b)),
            std::begin(c),
            comp);
        static_assert(std::is_same_v<decltype(res), decltype(std::begin(c))>);
      }
      { // First is shorter
        T a[]                    = {11, 33, 31, 41};
        T b[]                    = {22, 32, 43, 42, 52};
        T expected[]             = {11, 31};
        T c[std::size(expected)] = {};
        auto res                 = std::set_difference(
            policy,
            Iter1(std::begin(a)),
            Iter1(std::end(a)),
            Iter2(std::begin(b)),
            Iter2(std::end(b)),
            std::begin(c),
            comp);
        assert(res == std::end(c));
        assert(std::lexicographical_compare(std::begin(c), res, std::begin(expected), std::end(expected), comp) == 0);
      }
      { // First is longer
        T a[]                    = {22, 32, 43, 42, 52};
        T b[]                    = {11, 33, 31, 41};
        T expected[]             = {22, 42, 52};
        T c[std::size(expected)] = {};
        auto res                 = std::set_difference(
            policy,
            Iter1(std::begin(a)),
            Iter1(std::end(a)),
            Iter2(std::begin(b)),
            Iter2(std::end(b)),
            std::begin(c),
            comp);
        assert(res == std::end(c));
        assert(std::lexicographical_compare(std::begin(c), res, std::begin(expected), std::end(expected), comp) == 0);
      }
    }
  }
};

int main(int, char**) {
  types::for_each(
      types::forward_iterator_list<const TrivialSortableWithComp*>{}, types::apply_type_identity{[](auto v) {
        using Iter = typename decltype(v)::type;
        types::for_each(types::forward_iterator_list<const TrivialSortableWithComp*>{},
                        TestIteratorWithPolicies<types::partial_instantiation<Test, Iter>::template apply>{});
      }});
  return 0;
}
