//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++17

// UNSUPPORTED: libcpp-has-no-incomplete-pstl

// template <class ExecutionPolicy,
//           class ForwardIterator,
//           class Size>
//   void uninitialized_value_construct_n(ExecutionPolicy&& exec, ForwardIterator first, Size n);

#include <atomic>
#include <algorithm>
#include <cassert>
#include <cstring>
#include <functional>
#include <iterator>
#include <memory>
#include <numeric>

#include "test_execution_policies.h"
#include "test_iterators.h"
#include "test_macros.h"
#include "type_algorithms.h"
#include "runway_sample.h"

EXECUTION_POLICY_SFINAE_TEST(uninitialized_value_construct_n);

static_assert(sfinae_test_uninitialized_value_construct_n<int, int*, int>);
static_assert(!sfinae_test_uninitialized_value_construct_n<std::execution::parallel_policy, int*, int>);

// Each Counted object has a dedicated external atomic counter.
// To be able to connect with that counter, each Counted object knows that it is located inside a single pool.
struct Counted {
  static std::atomic_int counters[1073];
  static Counted* pool;
  std::atomic_int* counter;
  Counted() : counter(counters + (this - pool)) { counter->fetch_add(1); }
  Counted(Counted const& other) : counter(other.counter) { counter->fetch_add(1); }
  ~Counted() { counter->fetch_sub(1); }
  friend void operator&(Counted) = delete;
};
std::atomic_int Counted::counters[1073];
Counted* Counted::pool = nullptr;

template <class Iter>
struct TestCounted {
  template <class ExecutionPolicy>
  void operator()(ExecutionPolicy&& policy) {
    {
      std::atomic_int(&counters)[1073] = Counted::counters;
      Counted*& pool                   = Counted::pool;

      // initialize all Counted counters to 0
      std::fill_n(std::begin(counters), std::size(counters), 0);

      // Allocate memory for the Counted objects
      using Alloc = std::allocator<Counted>;
      Alloc alloc;
      pool = std::allocator_traits<Alloc>::allocate(alloc, std::size(counters));

      runway_sample(std::size(counters) + 1, [&](size_t size) {
        // Default-construct the Counted objects in range [0, size).
        std::uninitialized_value_construct_n(policy, Iter(pool), size);
        ASSERT_SAME_TYPE(decltype(std::uninitialized_value_construct_n(policy, Iter(pool), size)), void);

        // Verify that inside this range the counters are all 1 and outside the range they are all 0.
        assert(std::all_of(std::begin(counters), std::begin(counters) + size, [](auto& x) { return x == 1; }));
        assert(std::all_of(std::begin(counters) + size, std::end(counters), [](auto& x) { return x == 0; }));

        // Destroy the constructed objects and verify that all counters are 0.
        std::destroy(pool, pool + size);
        assert(std::all_of(std::begin(counters), std::end(counters), [](auto& x) { return x == 0; }));
      });

      std::allocator_traits<Alloc>::deallocate(alloc, pool, std::size(counters));
    }
  }
};

template <class Iter>
struct TestInt {
  template <class ExecutionPolicy>
  void operator()(ExecutionPolicy&& policy) {
    {
      constexpr int n = 1073;
      std::allocator<int> alloc;
      int* data = alloc.allocate(n);
      int* last = data + n;
      std::memset(data, 0xFF, n * sizeof(int));

      std::uninitialized_value_construct_n(policy, Iter(data), n);
      for (int i = 0; i != n; ++i) {
        assert(data[i] == 0);
      }

      std::destroy(data, last);
      alloc.deallocate(data, n);
    }
  }
};

int main(int, char**) {
  types::for_each(types::forward_iterator_list<Counted*>{}, TestIteratorWithPolicies<TestCounted>{});
  types::for_each(types::forward_iterator_list<int*>{}, TestIteratorWithPolicies<TestInt>{});
  return 0;
}
