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

// template<class ExecutionPolicy, class ForwardIterator, class Function>
//   void for_each(ExecutionPolicy&& exec,
//                 ForwardIterator first, ForwardIterator last,
//                 Function f);

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iterator>
#include <numeric>
#include <type_traits>
#include <vector>

#include "test_execution_policies.h"
#include "test_iterators.h"
#include "test_macros.h"
#include "type_algorithms.h"
#include "runway_sample.h"

EXECUTION_POLICY_SFINAE_TEST(for_each);

static_assert(sfinae_test_for_each<int, int*, int*, bool (*)(int)>);
static_assert(!sfinae_test_for_each<std::execution::parallel_policy, int*, int*, bool (*)(int)>);

struct NonCopyableNonMovable {
  NonCopyableNonMovable() = default;
  explicit NonCopyableNonMovable(int i) : i_(i) {}
  NonCopyableNonMovable(const NonCopyableNonMovable&)            = delete;
  NonCopyableNonMovable(NonCopyableNonMovable&&)                 = delete;
  NonCopyableNonMovable& operator=(const NonCopyableNonMovable&) = delete;
  NonCopyableNonMovable& operator=(NonCopyableNonMovable&&)      = delete;
  int value() const { return i_; }
  void visit() { ++visits_; }
  int visits() const { return visits_; }
  void reset() { visits_ = 0; }

private:
  int i_      = 0;
  int visits_ = 0;
};

template <class Iter>
struct Test {
  template <class ExecutionPolicy>
  void operator()(ExecutionPolicy&& policy) {
    { // Check the return type
      int a[] = {0};
      auto f  = [](int&) {};
      static_assert(std::is_same_v<decltype(std::for_each(policy, Iter(std::begin(a)), Iter(std::begin(a)), f)), void>);
    }
    { // Empty range, the function is never called
      int a[] = {42};
      std::for_each(policy, Iter(std::begin(a)), Iter(std::begin(a)), [](int&) { assert(false); });
      assert(a[0] == 42);
    }
    { // Single element
      int a[] = {42};
      std::for_each(policy, Iter(std::begin(a)), Iter(std::end(a)), [](int& v) { ++v; });
      assert(a[0] == 43);
    }
    { // Two elements
      int a[] = {1, 2};
      std::for_each(policy, Iter(std::begin(a)), Iter(std::end(a)), [](int& v) { v *= 10; });
      assert(a[0] == 10);
      assert(a[1] == 20);
    }
    { // Three elements
      int a[] = {1, 2, 3};
      std::for_each(policy, Iter(std::begin(a)), Iter(std::end(a)), [](int& v) { v *= 10; });
      assert(a[0] == 10);
      assert(a[1] == 20);
      assert(a[2] == 30);
    }
    { // Four elements, only a sub-range is processed
      int a[] = {1, 2, 3, 4};
      std::for_each(policy, Iter(std::begin(a) + 1), Iter(std::begin(a) + 3), [](int& v) { v = -v; });
      assert(a[0] == 1);
      assert(a[1] == -2);
      assert(a[2] == -3);
      assert(a[3] == 4);
    }
    { // The function can take its argument by value
      int a[]     = {1, 2, 3, 4};
      bool seen[] = {false, false, false, false};
      std::for_each(policy, Iter(std::begin(a)), Iter(std::end(a)), [&](int v) {
        assert(!seen[v - 1]);
        seen[v - 1] = true;
      });
      assert(std::all_of(std::begin(seen), std::end(seen), [](bool b) { return b; }));
    }
    { // Check that every single element in [0, i) is visited exactly once and the rest is untouched
      int a[1073];
      int visits[1073];
      runway_sample(std::size(a) + 1, [&](std::size_t i) {
        std::iota(std::begin(a), std::end(a), 0);
        std::fill(std::begin(visits), std::end(visits), 0);
        std::for_each(policy, Iter(std::begin(a)), Iter(std::begin(a) + i), [&](int& v) {
          ++visits[&v - std::begin(a)];
          v = -v;
        });
        for (std::size_t j = 0; j < std::size(a); ++j) {
          assert(visits[j] == (j < i ? 1 : 0));
          assert(a[j] == (j < i ? -static_cast<int>(j) : static_cast<int>(j)));
        }
      });
    }
  }
};

template <class Iter>
struct TestNonCopyableNonMovable {
  template <class ExecutionPolicy>
  void operator()(ExecutionPolicy&& policy) {
    { // Few elements
      NonCopyableNonMovable a[] = {NonCopyableNonMovable(1), NonCopyableNonMovable(5), NonCopyableNonMovable(7)};
      std::for_each(policy, Iter(std::begin(a)), Iter(std::end(a)), [](NonCopyableNonMovable& x) { x.visit(); });
      assert(a[0].visits() == 1 && a[1].visits() == 1 && a[2].visits() == 1);
      assert(a[0].value() == 1 && a[1].value() == 5 && a[2].value() == 7);
    }
    { // Many elements, check that every single element in [0, i) is visited exactly once
      NonCopyableNonMovable a[1073];
      runway_sample(std::size(a) + 1, [&](std::size_t i) {
        std::for_each(policy, Iter(std::begin(a)), Iter(std::begin(a) + i), [](NonCopyableNonMovable& x) {
          x.visit();
        });
        for (std::size_t j = 0; j < std::size(a); ++j) {
          assert(a[j].visits() == (j < i ? 1 : 0));
          a[j].reset();
        }
      });
    }
  }
};

int main(int, char**) {
  types::for_each(types::forward_iterator_list<int*>{}, TestIteratorWithPolicies<Test>{});
  types::for_each(
      types::forward_iterator_list<NonCopyableNonMovable*>{}, TestIteratorWithPolicies<TestNonCopyableNonMovable>{});
  return 0;
}
