//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// REQUIRES: std-at-least-c++17

// UNSUPPORTED: libcpp-has-no-incomplete-pstl

// template<class ExecutionPolicy, class ForwardIterator, class RandomAccessIterator,
//          class Compare>
//   RandomAccessIterator
//     partial_sort_copy(ExecutionPolicy&& exec,
//                       ForwardIterator first, ForwardIterator last,
//                       RandomAccessIterator result_first,
//                       RandomAccessIterator result_last,
//                       Compare comp);

#include <algorithm>
#include <cassert>
#include <functional>
#include <iterator>
#include <numeric>
#include <type_traits>
#include <random>
#include <vector>

#include "test_execution_policies.h"
#include "test_iterators.h"
#include "test_macros.h"
#include "type_algorithms.h"

EXECUTION_POLICY_SFINAE_TEST(partial_sort_copy);

static_assert(sfinae_test_partial_sort_copy<int, int*, int*, int*, int*, bool (*)(int, int)>);
static_assert(
    !sfinae_test_partial_sort_copy<std::execution::parallel_policy, int*, int*, int*, int*, bool (*)(int, int)>);

template <class Iter1, class Iter2>
struct Test {
  template <class ExecutionPolicy>
  void operator()(ExecutionPolicy&& policy) {
    auto comp = std::greater{};
    { // Check the return type
      int a[]  = {1};
      int r[]  = {0};
      auto ret = std::partial_sort_copy(
          policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(r)), Iter2(std::end(r)), comp);
      static_assert(std::is_same_v< decltype(ret), Iter2>);
    }
    { // Check empty input
      int a[] = {1};
      int r[] = {-1, -1};
      auto it = std::partial_sort_copy(
          policy, Iter1(std::begin(a)), Iter1(std::begin(a)), Iter2(std::begin(r)), Iter2(std::end(r)), comp);
      assert(it == Iter2(std::begin(r)));
      assert(r[0] == -1);
      assert(r[1] == -1);
    }
    { // Check empty output
      int a[] = {1, 2, 3};
      int r[] = {-1};
      auto it = std::partial_sort_copy(
          policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(r)), Iter2(std::begin(r)), comp);
      assert(it == Iter2(std::begin(r)));
      assert(r[0] == -1);
    }
    { // Check one element
      int a[] = {1};
      int r[] = {-1};
      auto it = std::partial_sort_copy(
          policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(r)), Iter2(std::end(r)), comp);
      assert(it == Iter2(std::end(r)));
      assert(r[0] == 1);
    }
    { // Check output smaller than input
      int a[] = {2, 1, 4, 4, 7, 2, 4, 1, 6};
      int r[] = {-1, -1, -1, -1};
      auto it = std::partial_sort_copy(
          policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(r)), Iter2(std::end(r)), comp);
      assert(it == Iter2(std::end(r)));
      assert(r[0] == 7);
      assert(r[1] == 6);
      assert(r[2] == 4);
      assert(r[3] == 4);
    }
    { // Check output the same size as input
      int a[]        = {2, 1, 4, 4, 7, 2, 4, 1, 6};
      int r[]        = {-1, -1, -1, -1, -1, -1, -1, -1, -1};
      int expected[] = {7, 6, 4, 4, 4, 2, 2, 1, 1};
      auto it        = std::partial_sort_copy(
          policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(r)), Iter2(std::end(r)), comp);
      assert(it == Iter2(std::end(r)));
      assert(std::equal(std::begin(r), std::end(r), std::begin(expected)));
    }
    { // Check output larger than input
      int a[] = {1, 3, 2};
      int r[] = {-1, -1, -1, -1, -1};
      auto it = std::partial_sort_copy(
          policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(r)), Iter2(std::end(r)), comp);
      assert(it == Iter2(std::begin(r) + 3));
      assert(r[0] == 3);
      assert(r[1] == 2);
      assert(r[2] == 1);
      assert(r[3] == -1);
      assert(r[4] == -1);
    }
    { // Check large input
#ifndef TEST_HAS_NO_RANDOM_DEVICE
      std::vector<int> a(1000);
      std::iota(std::begin(a), std::end(a), 0);
      std::shuffle(std::begin(a), std::end(a), std::mt19937_64(std::random_device{}()));
      std::vector<int> r(100, -1);
      auto it = std::partial_sort_copy(
          policy, Iter1(a.data()), Iter1(a.data() + a.size()), Iter2(r.data()), Iter2(r.data() + r.size()), comp);
      assert(it == Iter2(r.data() + r.size()));
      for (int i = 0; i != 100; ++i)
        assert(r[i] == 999 - i);
#endif
    }
  }
};

int main(int, char**) {
  types::for_each(types::forward_iterator_list<const int*>{}, types::apply_type_identity{[](auto v) {
                    using Iter = typename decltype(v)::type;
                    types::for_each(
                        types::random_access_iterator_list<int*>{},
                        TestIteratorWithPolicies<types::partial_instantiation<Test, Iter>::template apply>{});
                  }});

  return 0;
}
