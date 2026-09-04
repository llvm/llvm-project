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
//           class ForwardIterator>
//   void destroy(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last);

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

EXECUTION_POLICY_SFINAE_TEST(destroy);

static_assert(sfinae_test_destroy<int, int*, int*>);
static_assert(!sfinae_test_destroy<std::execution::parallel_policy, int*, int*>);

struct Counted {
  std::atomic_int* counter_;
  Counted(std::atomic_int* counter) : counter_(counter) { counter_->fetch_add(1); }
  Counted(Counted const& other) : counter_(other.counter_) { counter_->fetch_add(1); }
  ~Counted() { counter_->fetch_sub(1); }
  friend void operator&(Counted) = delete;
};

template <class Iter>
struct TestCounted {
  template <class ExecutionPolicy>
  void operator()(ExecutionPolicy&& policy) {
    {
      // Test destroy() with a range of Counted objects.
      // Ranges vary in size from 0 to 1073.
      // Each object has its own counter.
      std::atomic_int counters[1073];
      std::fill_n(std::begin(counters), std::size(counters), 0);

      using Alloc = std::allocator<Counted>;
      Alloc alloc;
      Counted* pool = std::allocator_traits<Alloc>::allocate(alloc, std::size(counters));

      runway_sample(std::size(counters) + 1, [&](size_t size) {
        for (std::size_t i = 0; i < size; ++i) {
          std::allocator_traits<Alloc>::construct(alloc, std::addressof(pool[i]), &counters[i]);
        }
        assert(std::all_of(std::begin(counters), std::begin(counters) + size, [](auto& x) { return x == 1; }));

        std::destroy(policy, Iter(pool), Iter(pool + size));
        ASSERT_SAME_TYPE(decltype(std::destroy(policy, Iter(pool), Iter(pool + size))), void);
        assert(std::all_of(std::begin(counters), std::begin(counters) + size, [](auto& x) { return x == 0; }));
      });

      std::allocator_traits<Alloc>::deallocate(alloc, pool, std::size(counters));
    }
  }
};

// std::destroy on a sequence of arrays is supported since C++20.
#if TEST_STD_VER >= 20
template <class Iter>
struct TestArrayCounted3 {
  template <class ExecutionPolicy>
  void operator()(ExecutionPolicy&& policy) {
    {
      // Test destroy() with a range of 5 Counted[3] objects.
      // A shared counter is used for all objects.
      using Array             = Counted[3];
      using Alloc             = std::allocator<Array>;
      std::atomic_int counter = 0;
      Alloc alloc;
      Array* pool = std::allocator_traits<Alloc>::allocate(alloc, 5);

      for (Array* p = pool; p != pool + 5; ++p) {
        Array& arr = *p;
        for (int i = 0; i != 3; ++i) {
          std::allocator_traits<Alloc>::construct(alloc, std::addressof(arr[i]), &counter);
        }
      }
      assert(counter == 5 * 3);

      std::destroy(policy, Iter(pool), Iter(pool + 5));
      ASSERT_SAME_TYPE(decltype(std::destroy(policy, Iter(pool), Iter(pool + 5))), void);
      assert(counter == 0);

      std::allocator_traits<Alloc>::deallocate(alloc, pool, 5);
    }
  }
};

template <class Iter>
struct TestArrayCounted3x2 {
  template <class ExecutionPolicy>
  void operator()(ExecutionPolicy&& policy) {
    {
      // Test destroy() with a range of 5 Counted[3][2] objects.
      // A shared counter is used for all objects.
      using Array             = Counted[3][2];
      using Alloc             = std::allocator<Array>;
      std::atomic_int counter = 0;
      Alloc alloc;
      Array* pool = std::allocator_traits<Alloc>::allocate(alloc, 5);

      for (Array* p = pool; p != pool + 5; ++p) {
        Array& arr = *p;
        for (int i = 0; i != 3; ++i) {
          for (int j = 0; j != 2; ++j) {
            std::allocator_traits<Alloc>::construct(alloc, std::addressof(arr[i][j]), &counter);
          }
        }
      }
      assert(counter == 5 * 3 * 2);

      std::destroy(policy, Iter(pool), Iter(pool + 5));
      ASSERT_SAME_TYPE(decltype(std::destroy(policy, Iter(pool), Iter(pool + 5))), void);
      assert(counter == 0);

      std::allocator_traits<Alloc>::deallocate(alloc, pool, 5);
    }
  }
};
#endif // TEST_STD_VER >= 20

int main(int, char**) {
  types::for_each(types::forward_iterator_list<Counted*>{}, TestIteratorWithPolicies<TestCounted>{});
#if TEST_STD_VER >= 20
  using CountedArray3 = Counted[3];
  types::for_each(types::forward_iterator_list<CountedArray3*>{}, TestIteratorWithPolicies<TestArrayCounted3>{});
  using CountedArray3x2 = Counted[3][2];
  types::for_each(types::forward_iterator_list<CountedArray3x2*>{}, TestIteratorWithPolicies<TestArrayCounted3x2>{});
#endif // TEST_STD_VER >= 20
  return 0;
}
