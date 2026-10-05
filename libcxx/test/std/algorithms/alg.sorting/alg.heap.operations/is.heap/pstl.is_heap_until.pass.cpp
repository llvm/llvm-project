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

// template<class ExecutionPolicy, class RandomAccessIterator>
//   RandomAccessIterator
//   is_heap_until(ExecutionPolicy&& exec,
//                 RandomAccessIterator first, RandomAccessIterator last);

#include <cstddef>
#include <algorithm>
#include <cassert>
#include <iterator>
#include <numeric>

#include "test_execution_policies.h"
#include "test_iterators.h"
#include "test_macros.h"
#include "type_algorithms.h"
#include "runway_sample.h"

EXECUTION_POLICY_SFINAE_TEST(is_heap_until);

static_assert(sfinae_test_is_heap_until<int, int*, int*>);
static_assert(!sfinae_test_is_heap_until<std::execution::parallel_policy, int*, int*>);

template <class Iter>
struct Test {
  template <class ExecutionPolicy>
  void operator()(ExecutionPolicy&& policy) {
    { // Check the return type
      int a[]  = {0};
      auto res = std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a)));
      static_assert(std::is_same_v<decltype(res), Iter>);
    }
    { // Empty
      int a[] = {0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::begin(a))) == Iter(std::begin(a)));
    }
    { // Single
      int a[] = {0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::end(a)));
    }
    { // Size=2, heap
      int a[] = {0, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::end(a)));
    }
    { // Size=2, not heap
      int a[] = {0, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 1));
    }
    { // Size=2, heap
      int a[] = {1, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::end(a)));
    }
    { // Size=3, heap
      int a[] = {0, 0, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::end(a)));
    }
    { // Size=3, not heap
      int a[] = {0, 0, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 2));
    }
    { // Size=3, not heap
      int a[] = {0, 1, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 1));
    }
    { // Size=3, not heap
      int a[] = {0, 1, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 1));
    }
    { // Size=3, heap
      int a[] = {1, 0, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::end(a)));
    }
    { // Size=3, heap
      int a[] = {1, 0, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::end(a)));
    }
    { // Size=3, heap
      int a[] = {1, 1, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::end(a)));
    }
    { // Size=4, heap
      int a[] = {0, 0, 0, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::end(a)));
    }
    { // Size=4, not heap
      int a[] = {0, 0, 0, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 3));
    }
    { // Size=4, not heap
      int a[] = {0, 0, 1, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 2));
    }
    { // Size=4, not heap
      int a[] = {0, 0, 1, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 2));
    }
    { // Size=4, not heap
      int a[] = {0, 1, 0, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 1));
    }
    { // Size=4, not heap
      int a[] = {0, 1, 0, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 1));
    }
    { // Size=4, not heap
      int a[] = {0, 1, 1, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 1));
    }
    { // Size=4, not heap
      int a[] = {0, 1, 1, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 1));
    }
    { // Size=4, heap
      int a[] = {1, 0, 0, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::end(a)));
    }
    { // Size=4, not heap
      int a[] = {1, 0, 0, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 3));
    }
    { // Size=4, heap
      int a[] = {1, 0, 1, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::end(a)));
    }
    { // Size=4, not heap
      int a[] = {1, 0, 1, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 3));
    }
    { // Size=4, heap
      int a[] = {1, 1, 0, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::end(a)));
    }
    { // Size=4, heap
      int a[] = {1, 1, 0, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::end(a)));
    }
    { // Size=4, heap
      int a[] = {1, 1, 1, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::end(a)));
    }
    { // Size=5, heap
      int a[] = {0, 0, 0, 0, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::end(a)));
    }
    { // Size=5, not heap
      int a[] = {0, 0, 0, 0, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 4));
    }
    { // Size=5, not heap
      int a[] = {0, 0, 0, 1, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 3));
    }
    { // Size=5, not heap
      int a[] = {0, 0, 0, 1, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 3));
    }
    { // Size=5, not heap
      int a[] = {0, 0, 1, 0, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 2));
    }
    { // Size=5, not heap
      int a[] = {0, 0, 1, 0, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 2));
    }
    { // Size=5, not heap
      int a[] = {0, 0, 1, 1, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 2));
    }
    { // Size=5, not heap
      int a[] = {0, 0, 1, 1, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 2));
    }
    { // Size=5, not heap
      int a[] = {0, 1, 0, 0, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 1));
    }
    { // Size=5, not heap
      int a[] = {0, 1, 0, 0, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 1));
    }
    { // Size=5, not heap
      int a[] = {0, 1, 0, 1, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 1));
    }
    { // Size=5, not heap
      int a[] = {0, 1, 0, 1, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 1));
    }
    { // Size=5, not heap
      int a[] = {0, 1, 1, 0, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 1));
    }
    { // Size=5, not heap
      int a[] = {0, 1, 1, 0, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 1));
    }
    { // Size=5, not heap
      int a[] = {0, 1, 1, 1, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 1));
    }
    { // Size=5, not heap
      int a[] = {0, 1, 1, 1, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 1));
    }
    { // Size=5, heap
      int a[] = {1, 0, 0, 0, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::end(a)));
    }
    { // Size=5, not heap
      int a[] = {1, 0, 0, 0, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 4));
    }
    { // Size=5, not heap
      int a[] = {1, 0, 0, 1, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 3));
    }
    { // Size=5, not heap
      int a[] = {1, 0, 0, 1, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 3));
    }
    { // Size=5, heap
      int a[] = {1, 0, 1, 0, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::end(a)));
    }
    { // Size=5, not heap
      int a[] = {1, 0, 1, 0, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 4));
    }
    { // Size=5, not heap
      int a[] = {1, 0, 1, 1, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 3));
    }
    { // Size=5, not heap
      int a[] = {1, 0, 1, 1, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 3));
    }
    { // Size=5, heap
      int a[] = {1, 1, 0, 0, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::end(a)));
    }
    { // Size=5, heap
      int a[] = {1, 1, 0, 0, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::end(a)));
    }
    { // Size=5, heap
      int a[] = {1, 1, 0, 1, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::end(a)));
    }
    { // Size=5, heap
      int a[] = {1, 1, 0, 1, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::end(a)));
    }
    { // Size=5, heap
      int a[] = {1, 1, 1, 0, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::end(a)));
    }
    { // Size=5, heap
      int a[] = {1, 1, 1, 0, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::end(a)));
    }
    { // Size=5, heap
      int a[] = {1, 1, 1, 1, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::end(a)));
    }
    { // Size=6, heap
      int a[] = {0, 0, 0, 0, 0, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::end(a)));
    }
    { // Size=6, not heap
      int a[] = {0, 0, 0, 0, 0, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 5));
    }
    { // Size=6, not heap
      int a[] = {0, 0, 0, 0, 1, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 4));
    }
    { // Size=6, not heap
      int a[] = {0, 0, 0, 0, 1, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 4));
    }
    { // Size=6, not heap
      int a[] = {0, 0, 0, 1, 0, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 3));
    }
    { // Size=6, not heap
      int a[] = {0, 0, 0, 1, 0, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 3));
    }
    { // Size=6, not heap
      int a[] = {0, 0, 0, 1, 1, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 3));
    }
    { // Size=6, not heap
      int a[] = {0, 0, 0, 1, 1, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 3));
    }
    { // Size=6, not heap
      int a[] = {0, 0, 1, 0, 0, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 2));
    }
    { // Size=6, not heap
      int a[] = {0, 0, 1, 0, 0, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 2));
    }
    { // Size=6, not heap
      int a[] = {0, 0, 1, 0, 1, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 2));
    }
    { // Size=6, not heap
      int a[] = {0, 0, 1, 0, 1, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 2));
    }
    { // Size=6, not heap
      int a[] = {0, 0, 1, 1, 0, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 2));
    }
    { // Size=6, not heap
      int a[] = {0, 0, 1, 1, 0, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 2));
    }
    { // Size=6, not heap
      int a[] = {0, 0, 1, 1, 1, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 2));
    }
    { // Size=6, not heap
      int a[] = {0, 0, 1, 1, 1, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 2));
    }
    { // Size=6, not heap
      int a[] = {0, 1, 0, 0, 0, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 1));
    }
    { // Size=6, not heap
      int a[] = {0, 1, 0, 0, 0, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 1));
    }
    { // Size=6, not heap
      int a[] = {0, 1, 0, 0, 1, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 1));
    }
    { // Size=6, not heap
      int a[] = {0, 1, 0, 0, 1, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 1));
    }
    { // Size=6, not heap
      int a[] = {0, 1, 0, 1, 0, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 1));
    }
    { // Size=6, not heap
      int a[] = {0, 1, 0, 1, 0, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 1));
    }
    { // Size=6, not heap
      int a[] = {0, 1, 0, 1, 1, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 1));
    }
    { // Size=6, not heap
      int a[] = {0, 1, 0, 1, 1, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 1));
    }
    { // Size=6, not heap
      int a[] = {0, 1, 1, 0, 0, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 1));
    }
    { // Size=6, not heap
      int a[] = {0, 1, 1, 0, 0, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 1));
    }
    { // Size=6, not heap
      int a[] = {0, 1, 1, 0, 1, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 1));
    }
    { // Size=6, not heap
      int a[] = {0, 1, 1, 0, 1, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 1));
    }
    { // Size=6, not heap
      int a[] = {0, 1, 1, 1, 0, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 1));
    }
    { // Size=6, not heap
      int a[] = {0, 1, 1, 1, 0, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 1));
    }
    { // Size=6, not heap
      int a[] = {0, 1, 1, 1, 1, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 1));
    }
    { // Size=6, not heap
      int a[] = {0, 1, 1, 1, 1, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 1));
    }
    { // Size=6, heap
      int a[] = {1, 0, 0, 0, 0, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::end(a)));
    }
    { // Size=6, not heap
      int a[] = {1, 0, 0, 0, 0, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 5));
    }
    { // Size=6, not heap
      int a[] = {1, 0, 0, 0, 1, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 4));
    }
    { // Size=6, not heap
      int a[] = {1, 0, 0, 0, 1, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 4));
    }
    { // Size=6, not heap
      int a[] = {1, 0, 0, 1, 0, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 3));
    }
    { // Size=6, not heap
      int a[] = {1, 0, 0, 1, 0, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 3));
    }
    { // Size=6, not heap
      int a[] = {1, 0, 0, 1, 1, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 3));
    }
    { // Size=6, not heap
      int a[] = {1, 0, 0, 1, 1, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 3));
    }
    { // Size=6, heap
      int a[] = {1, 0, 1, 0, 0, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::end(a)));
    }
    { // Size=6, heap
      int a[] = {1, 0, 1, 0, 0, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::end(a)));
    }
    { // Size=6, not heap
      int a[] = {1, 0, 1, 0, 1, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 4));
    }
    { // Size=6, not heap
      int a[] = {1, 0, 1, 0, 1, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 4));
    }
    { // Size=6, not heap
      int a[] = {1, 0, 1, 1, 0, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 3));
    }
    { // Size=6, not heap
      int a[] = {1, 0, 1, 1, 0, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 3));
    }
    { // Size=6, not heap
      int a[] = {1, 0, 1, 1, 1, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 3));
    }
    { // Size=6, not heap
      int a[] = {1, 0, 1, 1, 1, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 3));
    }
    { // Size=6, heap
      int a[] = {1, 1, 0, 0, 0, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::end(a)));
    }
    { // Size=6, not heap
      int a[] = {1, 1, 0, 0, 0, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 5));
    }
    { // Size=6, heap
      int a[] = {1, 1, 0, 0, 1, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::end(a)));
    }
    { // Size=6, not heap
      int a[] = {1, 1, 0, 0, 1, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 5));
    }
    { // Size=6, heap
      int a[] = {1, 1, 0, 1, 0, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::end(a)));
    }
    { // Size=6, not heap
      int a[] = {1, 1, 0, 1, 0, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 5));
    }
    { // Size=6, heap
      int a[] = {1, 1, 0, 1, 1, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::end(a)));
    }
    { // Size=6, not heap
      int a[] = {1, 1, 0, 1, 1, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 5));
    }
    { // Size=6, heap
      int a[] = {1, 1, 1, 0, 0, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::end(a)));
    }
    { // Size=6, heap
      int a[] = {1, 1, 1, 0, 0, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::end(a)));
    }
    { // Size=6, heap
      int a[] = {1, 1, 1, 0, 1, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::end(a)));
    }
    { // Size=6, heap
      int a[] = {1, 1, 1, 0, 1, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::end(a)));
    }
    { // Size=6, heap
      int a[] = {1, 1, 1, 1, 0, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::end(a)));
    }
    { // Size=6, heap
      int a[] = {1, 1, 1, 1, 0, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::end(a)));
    }
    { // Size=6, heap
      int a[] = {1, 1, 1, 1, 1, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::end(a)));
    }
    { // Size=7, heap
      int a[] = {0, 0, 0, 0, 0, 0, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::end(a)));
    }
    { // Size=7, not heap
      int a[] = {0, 0, 0, 0, 0, 0, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 6));
    }
    { // Size=7, not heap
      int a[] = {0, 0, 0, 0, 0, 1, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 5));
    }
    { // Size=7, not heap
      int a[] = {0, 0, 0, 0, 0, 1, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 5));
    }
    { // Size=7, not heap
      int a[] = {0, 0, 0, 0, 1, 0, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 4));
    }
    { // Size=7, not heap
      int a[] = {0, 0, 0, 0, 1, 0, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 4));
    }
    { // Size=7, not heap
      int a[] = {0, 0, 0, 0, 1, 1, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 4));
    }
    { // Size=7, not heap
      int a[] = {0, 0, 0, 0, 1, 1, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 4));
    }
    { // Size=7, not heap
      int a[] = {0, 0, 0, 1, 0, 0, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 3));
    }
    { // Size=7, not heap
      int a[] = {0, 0, 0, 1, 0, 0, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 3));
    }
    { // Size=7, not heap
      int a[] = {0, 0, 0, 1, 0, 1, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 3));
    }
    { // Size=7, not heap
      int a[] = {0, 0, 0, 1, 0, 1, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 3));
    }
    { // Size=7, not heap
      int a[] = {0, 0, 0, 1, 1, 0, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 3));
    }
    { // Size=7, not heap
      int a[] = {0, 0, 0, 1, 1, 0, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 3));
    }
    { // Size=7, not heap
      int a[] = {0, 0, 0, 1, 1, 1, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 3));
    }
    { // Size=7, not heap
      int a[] = {0, 0, 0, 1, 1, 1, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 3));
    }
    { // Size=7, not heap
      int a[] = {0, 0, 1, 0, 0, 0, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 2));
    }
    { // Size=7, not heap
      int a[] = {0, 0, 1, 0, 0, 0, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 2));
    }
    { // Size=7, not heap
      int a[] = {0, 0, 1, 0, 0, 1, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 2));
    }
    { // Size=7, not heap
      int a[] = {0, 0, 1, 0, 0, 1, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 2));
    }
    { // Size=7, not heap
      int a[] = {0, 0, 1, 0, 1, 0, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 2));
    }
    { // Size=7, not heap
      int a[] = {0, 0, 1, 0, 1, 0, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 2));
    }
    { // Size=7, not heap
      int a[] = {0, 0, 1, 0, 1, 1, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 2));
    }
    { // Size=7, not heap
      int a[] = {0, 0, 1, 0, 1, 1, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 2));
    }
    { // Size=7, not heap
      int a[] = {0, 0, 1, 1, 0, 0, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 2));
    }
    { // Size=7, not heap
      int a[] = {0, 0, 1, 1, 0, 0, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 2));
    }
    { // Size=7, not heap
      int a[] = {0, 0, 1, 1, 0, 1, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 2));
    }
    { // Size=7, not heap
      int a[] = {0, 0, 1, 1, 0, 1, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 2));
    }
    { // Size=7, not heap
      int a[] = {0, 0, 1, 1, 1, 0, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 2));
    }
    { // Size=7, not heap
      int a[] = {0, 0, 1, 1, 1, 0, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 2));
    }
    { // Size=7, not heap
      int a[] = {0, 0, 1, 1, 1, 1, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 2));
    }
    { // Size=7, not heap
      int a[] = {0, 0, 1, 1, 1, 1, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 2));
    }
    { // Size=7, not heap
      int a[] = {0, 1, 0, 0, 0, 0, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 1));
    }
    { // Size=7, not heap
      int a[] = {0, 1, 0, 0, 0, 0, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 1));
    }
    { // Size=7, not heap
      int a[] = {0, 1, 0, 0, 0, 1, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 1));
    }
    { // Size=7, not heap
      int a[] = {0, 1, 0, 0, 0, 1, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 1));
    }
    { // Size=7, not heap
      int a[] = {0, 1, 0, 0, 1, 0, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 1));
    }
    { // Size=7, not heap
      int a[] = {0, 1, 0, 0, 1, 0, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 1));
    }
    { // Size=7, not heap
      int a[] = {0, 1, 0, 0, 1, 1, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 1));
    }
    { // Size=7, not heap
      int a[] = {0, 1, 0, 0, 1, 1, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 1));
    }
    { // Size=7, not heap
      int a[] = {0, 1, 0, 1, 0, 0, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 1));
    }
    { // Size=7, not heap
      int a[] = {0, 1, 0, 1, 0, 0, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 1));
    }
    { // Size=7, not heap
      int a[] = {0, 1, 0, 1, 0, 1, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 1));
    }
    { // Size=7, not heap
      int a[] = {0, 1, 0, 1, 0, 1, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 1));
    }
    { // Size=7, not heap
      int a[] = {0, 1, 0, 1, 1, 0, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 1));
    }
    { // Size=7, not heap
      int a[] = {0, 1, 0, 1, 1, 0, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 1));
    }
    { // Size=7, not heap
      int a[] = {0, 1, 0, 1, 1, 1, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 1));
    }
    { // Size=7, not heap
      int a[] = {0, 1, 0, 1, 1, 1, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 1));
    }
    { // Size=7, not heap
      int a[] = {0, 1, 1, 0, 0, 0, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 1));
    }
    { // Size=7, not heap
      int a[] = {0, 1, 1, 0, 0, 0, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 1));
    }
    { // Size=7, not heap
      int a[] = {0, 1, 1, 0, 0, 1, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 1));
    }
    { // Size=7, not heap
      int a[] = {0, 1, 1, 0, 0, 1, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 1));
    }
    { // Size=7, not heap
      int a[] = {0, 1, 1, 0, 1, 0, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 1));
    }
    { // Size=7, not heap
      int a[] = {0, 1, 1, 0, 1, 0, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 1));
    }
    { // Size=7, not heap
      int a[] = {0, 1, 1, 0, 1, 1, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 1));
    }
    { // Size=7, not heap
      int a[] = {0, 1, 1, 0, 1, 1, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 1));
    }
    { // Size=7, not heap
      int a[] = {0, 1, 1, 1, 0, 0, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 1));
    }
    { // Size=7, not heap
      int a[] = {0, 1, 1, 1, 0, 0, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 1));
    }
    { // Size=7, not heap
      int a[] = {0, 1, 1, 1, 0, 1, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 1));
    }
    { // Size=7, not heap
      int a[] = {0, 1, 1, 1, 0, 1, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 1));
    }
    { // Size=7, not heap
      int a[] = {0, 1, 1, 1, 1, 0, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 1));
    }
    { // Size=7, not heap
      int a[] = {0, 1, 1, 1, 1, 0, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 1));
    }
    { // Size=7, not heap
      int a[] = {0, 1, 1, 1, 1, 1, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 1));
    }
    { // Size=7, not heap
      int a[] = {0, 1, 1, 1, 1, 1, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 1));
    }
    { // Size=7, heap
      int a[] = {1, 0, 0, 0, 0, 0, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::end(a)));
    }
    { // Size=7, not heap
      int a[] = {1, 0, 0, 0, 0, 0, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 6));
    }
    { // Size=7, not heap
      int a[] = {1, 0, 0, 0, 0, 1, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 5));
    }
    { // Size=7, not heap
      int a[] = {1, 0, 0, 0, 0, 1, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 5));
    }
    { // Size=7, not heap
      int a[] = {1, 0, 0, 0, 1, 0, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 4));
    }
    { // Size=7, not heap
      int a[] = {1, 0, 0, 0, 1, 0, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 4));
    }
    { // Size=7, not heap
      int a[] = {1, 0, 0, 0, 1, 1, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 4));
    }
    { // Size=7, not heap
      int a[] = {1, 0, 0, 0, 1, 1, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 4));
    }
    { // Size=7, not heap
      int a[] = {1, 0, 0, 1, 0, 0, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 3));
    }
    { // Size=7, not heap
      int a[] = {1, 0, 0, 1, 0, 0, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 3));
    }
    { // Size=7, not heap
      int a[] = {1, 0, 0, 1, 0, 1, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 3));
    }
    { // Size=7, not heap
      int a[] = {1, 0, 0, 1, 0, 1, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 3));
    }
    { // Size=7, not heap
      int a[] = {1, 0, 0, 1, 1, 0, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 3));
    }
    { // Size=7, not heap
      int a[] = {1, 0, 0, 1, 1, 0, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 3));
    }
    { // Size=7, not heap
      int a[] = {1, 0, 0, 1, 1, 1, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 3));
    }
    { // Size=7, not heap
      int a[] = {1, 0, 0, 1, 1, 1, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 3));
    }
    { // Size=7, heap
      int a[] = {1, 0, 1, 0, 0, 0, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::end(a)));
    }
    { // Size=7, heap
      int a[] = {1, 0, 1, 0, 0, 0, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::end(a)));
    }
    { // Size=7, heap
      int a[] = {1, 0, 1, 0, 0, 1, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::end(a)));
    }
    { // Size=7, heap
      int a[] = {1, 0, 1, 0, 0, 1, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::end(a)));
    }
    { // Size=7, not heap
      int a[] = {1, 0, 1, 0, 1, 0, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 4));
    }
    { // Size=7, not heap
      int a[] = {1, 0, 1, 0, 1, 0, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 4));
    }
    { // Size=7, not heap
      int a[] = {1, 0, 1, 0, 1, 1, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 4));
    }
    { // Size=7, not heap
      int a[] = {1, 0, 1, 0, 1, 1, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 4));
    }
    { // Size=7, not heap
      int a[] = {1, 0, 1, 1, 0, 0, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 3));
    }
    { // Size=7, not heap
      int a[] = {1, 0, 1, 1, 0, 0, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 3));
    }
    { // Size=7, not heap
      int a[] = {1, 0, 1, 1, 0, 1, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 3));
    }
    { // Size=7, not heap
      int a[] = {1, 0, 1, 1, 0, 1, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 3));
    }
    { // Size=7, not heap
      int a[] = {1, 0, 1, 1, 1, 0, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 3));
    }
    { // Size=7, not heap
      int a[] = {1, 0, 1, 1, 1, 0, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 3));
    }
    { // Size=7, not heap
      int a[] = {1, 0, 1, 1, 1, 1, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 3));
    }
    { // Size=7, not heap
      int a[] = {1, 0, 1, 1, 1, 1, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 3));
    }
    { // Size=7, heap
      int a[] = {1, 1, 0, 0, 0, 0, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::end(a)));
    }
    { // Size=7, not heap
      int a[] = {1, 1, 0, 0, 0, 0, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 6));
    }
    { // Size=7, not heap
      int a[] = {1, 1, 0, 0, 0, 1, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 5));
    }
    { // Size=7, not heap
      int a[] = {1, 1, 0, 0, 0, 1, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 5));
    }
    { // Size=7, heap
      int a[] = {1, 1, 0, 0, 1, 0, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::end(a)));
    }
    { // Size=7, not heap
      int a[] = {1, 1, 0, 0, 1, 0, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 6));
    }
    { // Size=7, not heap
      int a[] = {1, 1, 0, 0, 1, 1, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 5));
    }
    { // Size=7, not heap
      int a[] = {1, 1, 0, 0, 1, 1, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 5));
    }
    { // Size=7, heap
      int a[] = {1, 1, 0, 1, 0, 0, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::end(a)));
    }
    { // Size=7, not heap
      int a[] = {1, 1, 0, 1, 0, 0, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 6));
    }
    { // Size=7, not heap
      int a[] = {1, 1, 0, 1, 0, 1, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 5));
    }
    { // Size=7, not heap
      int a[] = {1, 1, 0, 1, 0, 1, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 5));
    }
    { // Size=7, heap
      int a[] = {1, 1, 0, 1, 1, 0, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::end(a)));
    }
    { // Size=7, not heap
      int a[] = {1, 1, 0, 1, 1, 0, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 6));
    }
    { // Size=7, not heap
      int a[] = {1, 1, 0, 1, 1, 1, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 5));
    }
    { // Size=7, not heap
      int a[] = {1, 1, 0, 1, 1, 1, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 5));
    }
    { // Size=7, heap
      int a[] = {1, 1, 1, 0, 0, 0, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::end(a)));
    }
    { // Size=7, heap
      int a[] = {1, 1, 1, 0, 0, 0, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::end(a)));
    }
    { // Size=7, heap
      int a[] = {1, 1, 1, 0, 0, 1, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::end(a)));
    }
    { // Size=7, heap
      int a[] = {1, 1, 1, 0, 0, 1, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::end(a)));
    }
    { // Size=7, heap
      int a[] = {1, 1, 1, 0, 1, 0, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::end(a)));
    }
    { // Size=7, heap
      int a[] = {1, 1, 1, 0, 1, 0, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::end(a)));
    }
    { // Size=7, heap
      int a[] = {1, 1, 1, 0, 1, 1, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::end(a)));
    }
    { // Size=7, heap
      int a[] = {1, 1, 1, 0, 1, 1, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::end(a)));
    }
    { // Size=7, heap
      int a[] = {1, 1, 1, 1, 0, 0, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::end(a)));
    }
    { // Size=7, heap
      int a[] = {1, 1, 1, 1, 0, 0, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::end(a)));
    }
    { // Size=7, heap
      int a[] = {1, 1, 1, 1, 0, 1, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::end(a)));
    }
    { // Size=7, heap
      int a[] = {1, 1, 1, 1, 0, 1, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::end(a)));
    }
    { // Size=7, heap
      int a[] = {1, 1, 1, 1, 1, 0, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::end(a)));
    }
    { // Size=7, heap
      int a[] = {1, 1, 1, 1, 1, 0, 1};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::end(a)));
    }
    { // Size=7, heap
      int a[] = {1, 1, 1, 1, 1, 1, 0};
      assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::end(a)));
    }
    { // Break a heap at sampled locations
      int a[1073];
      std::iota(std::begin(a), std::end(a), 0);
      std::make_heap(std::begin(a), std::end(a));
      runway_sample(std::size(a), [&](size_t i) {
        if (i == 0)
          return;
        int old = std::exchange(a[i], 10000);
        assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + i));
        a[i] = old;
      });
    }
    { // Check valid sub-heaps at sampled locations (to force different partitions)
      int a[1073];
      std::iota(std::begin(a), std::end(a), 0);
      std::make_heap(std::begin(a), std::end(a));
      runway_sample(std::size(a), [&](size_t i) {
        assert(std::is_heap_until(policy, Iter(std::begin(a)), Iter(std::begin(a) + i)) == Iter(std::begin(a) + i));
      });
    }
  }
};

int main(int, char**) {
  types::for_each(types::random_access_iterator_list<const int*>{}, TestIteratorWithPolicies<Test>{});
  return 0;
}
