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
//           class T>
//   void uninitialized_fill(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last, const T& value);

#include <atomic>
#include <algorithm>
#include <cassert>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>

#include "test_execution_policies.h"
#include "test_iterators.h"
#include "test_macros.h"
#include "type_algorithms.h"
#include "runway_sample.h"

EXECUTION_POLICY_SFINAE_TEST(uninitialized_fill);

static_assert(sfinae_test_uninitialized_fill<int, int*, int*, int>);
static_assert(!sfinae_test_uninitialized_fill<std::execution::parallel_policy, int*, int*, int>);

// Init payload to initialize each Counted object with a pointer to its dedicated counter
struct CountedInit {
  struct Counted* first_;
  std::atomic_int* counters_;
};

// Each Counted object has a dedicated external atomic counter
struct Counted {
  std::atomic_int* counter_;
  explicit Counted(CountedInit init) : counter_(init.counters_ + (this - init.first_)) { counter_->fetch_add(1); }
  Counted(Counted const& other) : counter_(other.counter_) { counter_->fetch_add(1); }
  ~Counted() { counter_->fetch_sub(1); }
  friend void operator&(Counted) = delete;
};

template <class Iter>
struct TestCounted {
  template <class ExecutionPolicy>
  void operator()(ExecutionPolicy&& policy) {
    {
      // Test uninitialized_fill() with a range of Counted objects.
      // Ranges vary in size from 0 to 1073.
      // Each object has its own counter.
      std::atomic_int counters[1073];
      std::fill_n(std::begin(counters), std::size(counters), 0); // initialize all counters to 0

      // Allocate memory for the Counted objects
      using Alloc = std::allocator<Counted>;
      Alloc alloc;
      Counted* pool = std::allocator_traits<Alloc>::allocate(alloc, std::size(counters));

      runway_sample(std::size(counters) + 1, [&](size_t size) {
        // Construct the Counted object in range [0, size).
        std::uninitialized_fill(policy, Iter(pool), Iter(pool + size), CountedInit{pool, counters});
        ASSERT_SAME_TYPE(
            decltype(std::uninitialized_fill(policy, Iter(pool), Iter(pool + size), CountedInit{pool, counters})),
            void);

        // Verify that inside this range the counters are all 1 and outside the range they are all 0.
        assert(std::all_of(std::begin(counters), std::begin(counters) + size, [](auto& x) { return x == 1; }));
        assert(std::all_of(std::begin(counters) + size, std::end(counters), [](auto& x) { return x == 0; }));

        // Destroy the constructed objects and verify that all counters are 0.
        std::destroy(Iter(pool), Iter(pool + size));
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

      std::uninitialized_fill(policy, Iter(data), Iter(last), 42);
      for (int i = 0; i != n; ++i) {
        assert(data[i] == 42);
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
