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

// template<class ExecutionPolicy, class RandomAccessIterator, class Compare>
//   bool
//   is_heap(ExecutionPolicy&& exec,
//           RandomAccessIterator first, RandomAccessIterator last,
//           Compare comp);

#include <cstddef>
#include <algorithm>
#include <cassert>
#include <functional>
#include <iterator>
#include <numeric>

#include "test_execution_policies.h"
#include "test_iterators.h"
#include "test_macros.h"
#include "type_algorithms.h"
#include "runway_sample.h"

EXECUTION_POLICY_SFINAE_TEST(is_heap);

static_assert(sfinae_test_is_heap<int, int*, int*, bool (*)(int, int)>);
static_assert(!sfinae_test_is_heap<std::execution::parallel_policy, int*, int*, bool (*)(int, int)>);

template <class Iter>
struct Test {
  template <class ExecutionPolicy>
  void operator()(ExecutionPolicy&& policy) {
    std::greater<int> comp;
    { // Check the return type
      int a[]  = {0};
      auto res = std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp);
      static_assert(std::is_same_v<decltype(res), bool>);
    }
    { // Empty
      int a[] = {0};
      assert(std::is_heap(policy, Iter(std::begin(a)), Iter(std::begin(a)), comp));
    }
    { // Single
      int a[] = {0};
      assert(std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=2, heap
      int a[] = {0, 0};
      assert(std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=2, heap
      int a[] = {0, 1};
      assert(std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=2, not heap
      int a[] = {1, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=3, heap
      int a[] = {0, 0, 0};
      assert(std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=3, heap
      int a[] = {0, 0, 1};
      assert(std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=3, heap
      int a[] = {0, 1, 0};
      assert(std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=3, heap
      int a[] = {0, 1, 1};
      assert(std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=3, not heap
      int a[] = {1, 0, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=3, not heap
      int a[] = {1, 0, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=3, not heap
      int a[] = {1, 1, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=4, heap
      int a[] = {0, 0, 0, 0};
      assert(std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=4, heap
      int a[] = {0, 0, 0, 1};
      assert(std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=4, heap
      int a[] = {0, 0, 1, 0};
      assert(std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=4, heap
      int a[] = {0, 0, 1, 1};
      assert(std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=4, not heap
      int a[] = {0, 1, 0, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=4, heap
      int a[] = {0, 1, 0, 1};
      assert(std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=4, not heap
      int a[] = {0, 1, 1, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=4, heap
      int a[] = {0, 1, 1, 1};
      assert(std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=4, not heap
      int a[] = {1, 0, 0, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=4, not heap
      int a[] = {1, 0, 0, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=4, not heap
      int a[] = {1, 0, 1, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=4, not heap
      int a[] = {1, 0, 1, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=4, not heap
      int a[] = {1, 1, 0, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=4, not heap
      int a[] = {1, 1, 0, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=4, not heap
      int a[] = {1, 1, 1, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=5, heap
      int a[] = {0, 0, 0, 0, 0};
      assert(std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=5, heap
      int a[] = {0, 0, 0, 0, 1};
      assert(std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=5, heap
      int a[] = {0, 0, 0, 1, 0};
      assert(std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=5, heap
      int a[] = {0, 0, 0, 1, 1};
      assert(std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=5, heap
      int a[] = {0, 0, 1, 0, 0};
      assert(std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=5, heap
      int a[] = {0, 0, 1, 0, 1};
      assert(std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=5, heap
      int a[] = {0, 0, 1, 1, 0};
      assert(std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=5, heap
      int a[] = {0, 0, 1, 1, 1};
      assert(std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=5, not heap
      int a[] = {0, 1, 0, 0, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=5, not heap
      int a[] = {0, 1, 0, 0, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=5, not heap
      int a[] = {0, 1, 0, 1, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=5, heap
      int a[] = {0, 1, 0, 1, 1};
      assert(std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=5, not heap
      int a[] = {0, 1, 1, 0, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=5, not heap
      int a[] = {0, 1, 1, 0, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=5, not heap
      int a[] = {0, 1, 1, 1, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=5, heap
      int a[] = {0, 1, 1, 1, 1};
      assert(std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=5, not heap
      int a[] = {1, 0, 0, 0, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=5, not heap
      int a[] = {1, 0, 0, 0, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=5, not heap
      int a[] = {1, 0, 0, 1, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=5, not heap
      int a[] = {1, 0, 0, 1, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=5, not heap
      int a[] = {1, 0, 1, 0, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=5, not heap
      int a[] = {1, 0, 1, 0, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=5, not heap
      int a[] = {1, 0, 1, 1, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=5, not heap
      int a[] = {1, 0, 1, 1, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=5, not heap
      int a[] = {1, 1, 0, 0, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=5, not heap
      int a[] = {1, 1, 0, 0, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=5, not heap
      int a[] = {1, 1, 0, 1, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=5, not heap
      int a[] = {1, 1, 0, 1, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=5, not heap
      int a[] = {1, 1, 1, 0, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=5, not heap
      int a[] = {1, 1, 1, 0, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=5, not heap
      int a[] = {1, 1, 1, 1, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=6, heap
      int a[] = {0, 0, 0, 0, 0, 0};
      assert(std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=6, heap
      int a[] = {0, 0, 0, 0, 0, 1};
      assert(std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=6, heap
      int a[] = {0, 0, 0, 0, 1, 0};
      assert(std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=6, heap
      int a[] = {0, 0, 0, 0, 1, 1};
      assert(std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=6, heap
      int a[] = {0, 0, 0, 1, 0, 0};
      assert(std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=6, heap
      int a[] = {0, 0, 0, 1, 0, 1};
      assert(std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=6, heap
      int a[] = {0, 0, 0, 1, 1, 0};
      assert(std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=6, heap
      int a[] = {0, 0, 0, 1, 1, 1};
      assert(std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=6, not heap
      int a[] = {0, 0, 1, 0, 0, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=6, heap
      int a[] = {0, 0, 1, 0, 0, 1};
      assert(std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=6, not heap
      int a[] = {0, 0, 1, 0, 1, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=6, heap
      int a[] = {0, 0, 1, 0, 1, 1};
      assert(std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=6, not heap
      int a[] = {0, 0, 1, 1, 0, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=6, heap
      int a[] = {0, 0, 1, 1, 0, 1};
      assert(std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=6, not heap
      int a[] = {0, 0, 1, 1, 1, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=6, heap
      int a[] = {0, 0, 1, 1, 1, 1};
      assert(std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=6, not heap
      int a[] = {0, 1, 0, 0, 0, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=6, not heap
      int a[] = {0, 1, 0, 0, 0, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=6, not heap
      int a[] = {0, 1, 0, 0, 1, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=6, not heap
      int a[] = {0, 1, 0, 0, 1, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=6, not heap
      int a[] = {0, 1, 0, 1, 0, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=6, not heap
      int a[] = {0, 1, 0, 1, 0, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=6, heap
      int a[] = {0, 1, 0, 1, 1, 0};
      assert(std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=6, heap
      int a[] = {0, 1, 0, 1, 1, 1};
      assert(std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=6, not heap
      int a[] = {0, 1, 1, 0, 0, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=6, not heap
      int a[] = {0, 1, 1, 0, 0, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=6, not heap
      int a[] = {0, 1, 1, 0, 1, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=6, not heap
      int a[] = {0, 1, 1, 0, 1, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=6, not heap
      int a[] = {0, 1, 1, 1, 0, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=6, not heap
      int a[] = {0, 1, 1, 1, 0, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=6, not heap
      int a[] = {0, 1, 1, 1, 1, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=6, heap
      int a[] = {0, 1, 1, 1, 1, 1};
      assert(std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=6, not heap
      int a[] = {1, 0, 0, 0, 0, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=6, not heap
      int a[] = {1, 0, 0, 0, 0, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=6, not heap
      int a[] = {1, 0, 0, 0, 1, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=6, not heap
      int a[] = {1, 0, 0, 0, 1, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=6, not heap
      int a[] = {1, 0, 0, 1, 0, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=6, not heap
      int a[] = {1, 0, 0, 1, 0, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=6, not heap
      int a[] = {1, 0, 0, 1, 1, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=6, not heap
      int a[] = {1, 0, 0, 1, 1, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=6, not heap
      int a[] = {1, 0, 1, 0, 0, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=6, not heap
      int a[] = {1, 0, 1, 0, 0, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=6, not heap
      int a[] = {1, 0, 1, 0, 1, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=6, not heap
      int a[] = {1, 0, 1, 0, 1, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=6, not heap
      int a[] = {1, 0, 1, 1, 0, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=6, not heap
      int a[] = {1, 0, 1, 1, 0, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=6, not heap
      int a[] = {1, 0, 1, 1, 1, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=6, not heap
      int a[] = {1, 0, 1, 1, 1, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=6, not heap
      int a[] = {1, 1, 0, 0, 0, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=6, not heap
      int a[] = {1, 1, 0, 0, 0, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=6, not heap
      int a[] = {1, 1, 0, 0, 1, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=6, not heap
      int a[] = {1, 1, 0, 0, 1, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=6, not heap
      int a[] = {1, 1, 0, 1, 0, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=6, not heap
      int a[] = {1, 1, 0, 1, 0, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=6, not heap
      int a[] = {1, 1, 0, 1, 1, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=6, not heap
      int a[] = {1, 1, 0, 1, 1, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=6, not heap
      int a[] = {1, 1, 1, 0, 0, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=6, not heap
      int a[] = {1, 1, 1, 0, 0, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=6, not heap
      int a[] = {1, 1, 1, 0, 1, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=6, not heap
      int a[] = {1, 1, 1, 0, 1, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=6, not heap
      int a[] = {1, 1, 1, 1, 0, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=6, not heap
      int a[] = {1, 1, 1, 1, 0, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=6, not heap
      int a[] = {1, 1, 1, 1, 1, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, heap
      int a[] = {0, 0, 0, 0, 0, 0, 0};
      assert(std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, heap
      int a[] = {0, 0, 0, 0, 0, 0, 1};
      assert(std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, heap
      int a[] = {0, 0, 0, 0, 0, 1, 0};
      assert(std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, heap
      int a[] = {0, 0, 0, 0, 0, 1, 1};
      assert(std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, heap
      int a[] = {0, 0, 0, 0, 1, 0, 0};
      assert(std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, heap
      int a[] = {0, 0, 0, 0, 1, 0, 1};
      assert(std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, heap
      int a[] = {0, 0, 0, 0, 1, 1, 0};
      assert(std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, heap
      int a[] = {0, 0, 0, 0, 1, 1, 1};
      assert(std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, heap
      int a[] = {0, 0, 0, 1, 0, 0, 0};
      assert(std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, heap
      int a[] = {0, 0, 0, 1, 0, 0, 1};
      assert(std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, heap
      int a[] = {0, 0, 0, 1, 0, 1, 0};
      assert(std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, heap
      int a[] = {0, 0, 0, 1, 0, 1, 1};
      assert(std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, heap
      int a[] = {0, 0, 0, 1, 1, 0, 0};
      assert(std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, heap
      int a[] = {0, 0, 0, 1, 1, 0, 1};
      assert(std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, heap
      int a[] = {0, 0, 0, 1, 1, 1, 0};
      assert(std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, heap
      int a[] = {0, 0, 0, 1, 1, 1, 1};
      assert(std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {0, 0, 1, 0, 0, 0, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {0, 0, 1, 0, 0, 0, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {0, 0, 1, 0, 0, 1, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, heap
      int a[] = {0, 0, 1, 0, 0, 1, 1};
      assert(std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {0, 0, 1, 0, 1, 0, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {0, 0, 1, 0, 1, 0, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {0, 0, 1, 0, 1, 1, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, heap
      int a[] = {0, 0, 1, 0, 1, 1, 1};
      assert(std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {0, 0, 1, 1, 0, 0, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {0, 0, 1, 1, 0, 0, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {0, 0, 1, 1, 0, 1, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, heap
      int a[] = {0, 0, 1, 1, 0, 1, 1};
      assert(std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {0, 0, 1, 1, 1, 0, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {0, 0, 1, 1, 1, 0, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {0, 0, 1, 1, 1, 1, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, heap
      int a[] = {0, 0, 1, 1, 1, 1, 1};
      assert(std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {0, 1, 0, 0, 0, 0, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {0, 1, 0, 0, 0, 0, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {0, 1, 0, 0, 0, 1, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {0, 1, 0, 0, 0, 1, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {0, 1, 0, 0, 1, 0, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {0, 1, 0, 0, 1, 0, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {0, 1, 0, 0, 1, 1, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {0, 1, 0, 0, 1, 1, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {0, 1, 0, 1, 0, 0, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {0, 1, 0, 1, 0, 0, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {0, 1, 0, 1, 0, 1, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {0, 1, 0, 1, 0, 1, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, heap
      int a[] = {0, 1, 0, 1, 1, 0, 0};
      assert(std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, heap
      int a[] = {0, 1, 0, 1, 1, 0, 1};
      assert(std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, heap
      int a[] = {0, 1, 0, 1, 1, 1, 0};
      assert(std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, heap
      int a[] = {0, 1, 0, 1, 1, 1, 1};
      assert(std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {0, 1, 1, 0, 0, 0, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {0, 1, 1, 0, 0, 0, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {0, 1, 1, 0, 0, 1, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {0, 1, 1, 0, 0, 1, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {0, 1, 1, 0, 1, 0, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {0, 1, 1, 0, 1, 0, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {0, 1, 1, 0, 1, 1, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {0, 1, 1, 0, 1, 1, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {0, 1, 1, 1, 0, 0, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {0, 1, 1, 1, 0, 0, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {0, 1, 1, 1, 0, 1, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {0, 1, 1, 1, 0, 1, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {0, 1, 1, 1, 1, 0, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {0, 1, 1, 1, 1, 0, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {0, 1, 1, 1, 1, 1, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, heap
      int a[] = {0, 1, 1, 1, 1, 1, 1};
      assert(std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {1, 0, 0, 0, 0, 0, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {1, 0, 0, 0, 0, 0, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {1, 0, 0, 0, 0, 1, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {1, 0, 0, 0, 0, 1, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {1, 0, 0, 0, 1, 0, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {1, 0, 0, 0, 1, 0, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {1, 0, 0, 0, 1, 1, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {1, 0, 0, 0, 1, 1, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {1, 0, 0, 1, 0, 0, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {1, 0, 0, 1, 0, 0, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {1, 0, 0, 1, 0, 1, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {1, 0, 0, 1, 0, 1, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {1, 0, 0, 1, 1, 0, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {1, 0, 0, 1, 1, 0, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {1, 0, 0, 1, 1, 1, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {1, 0, 0, 1, 1, 1, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {1, 0, 1, 0, 0, 0, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {1, 0, 1, 0, 0, 0, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {1, 0, 1, 0, 0, 1, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {1, 0, 1, 0, 0, 1, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {1, 0, 1, 0, 1, 0, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {1, 0, 1, 0, 1, 0, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {1, 0, 1, 0, 1, 1, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {1, 0, 1, 0, 1, 1, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {1, 0, 1, 1, 0, 0, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {1, 0, 1, 1, 0, 0, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {1, 0, 1, 1, 0, 1, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {1, 0, 1, 1, 0, 1, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {1, 0, 1, 1, 1, 0, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {1, 0, 1, 1, 1, 0, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {1, 0, 1, 1, 1, 1, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {1, 0, 1, 1, 1, 1, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {1, 1, 0, 0, 0, 0, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {1, 1, 0, 0, 0, 0, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {1, 1, 0, 0, 0, 1, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {1, 1, 0, 0, 0, 1, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {1, 1, 0, 0, 1, 0, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {1, 1, 0, 0, 1, 0, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {1, 1, 0, 0, 1, 1, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {1, 1, 0, 0, 1, 1, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {1, 1, 0, 1, 0, 0, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {1, 1, 0, 1, 0, 0, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {1, 1, 0, 1, 0, 1, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {1, 1, 0, 1, 0, 1, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {1, 1, 0, 1, 1, 0, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {1, 1, 0, 1, 1, 0, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {1, 1, 0, 1, 1, 1, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {1, 1, 0, 1, 1, 1, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {1, 1, 1, 0, 0, 0, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {1, 1, 1, 0, 0, 0, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {1, 1, 1, 0, 0, 1, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {1, 1, 1, 0, 0, 1, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {1, 1, 1, 0, 1, 0, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {1, 1, 1, 0, 1, 0, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {1, 1, 1, 0, 1, 1, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {1, 1, 1, 0, 1, 1, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {1, 1, 1, 1, 0, 0, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {1, 1, 1, 1, 0, 0, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {1, 1, 1, 1, 0, 1, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {1, 1, 1, 1, 0, 1, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {1, 1, 1, 1, 1, 0, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {1, 1, 1, 1, 1, 0, 1};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Size=7, not heap
      int a[] = {1, 1, 1, 1, 1, 1, 0};
      assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
    }
    { // Break a heap at sampled locations
      int a[1073];
      std::iota(std::begin(a), std::end(a), 0);
      std::make_heap(std::begin(a), std::end(a), comp);
      runway_sample(std::size(a), [&](size_t i) {
        if (i == 0)
          return;
        int old = std::exchange(a[i], -10000);
        assert(!std::is_heap(policy, Iter(std::begin(a)), Iter(std::end(a)), comp));
        a[i] = old;
      });
    }
    { // Check valid sub-heaps at sampled locations (to force different partitions)
      int a[1073];
      std::iota(std::begin(a), std::end(a), 0);
      std::make_heap(std::begin(a), std::end(a), comp);
      runway_sample(std::size(a), [&](size_t i) {
        assert(std::is_heap(policy, Iter(std::begin(a)), Iter(std::begin(a) + i), comp));
      });
    }
  }
};

int main(int, char**) {
  types::for_each(types::random_access_iterator_list<const int*>{}, TestIteratorWithPolicies<Test>{});
  return 0;
}
