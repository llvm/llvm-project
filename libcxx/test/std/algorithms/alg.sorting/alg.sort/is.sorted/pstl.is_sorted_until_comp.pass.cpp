//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++17

// UNSUPPORTED: libcpp-has-no-incomplete-pstl

// template<class ExecutionPolicy, class ForwardIterator, class Comp>
// ForwardIterator is_sorted_until(ExecutionPolicy&& exec,
//                                 ForwardIterator first, ForwardIterator last,
//                                 Comp comp);

#include <algorithm>
#include <cassert>
#include <functional>
#include <iterator>
#include <limits>
#include <numeric>

#include "test_execution_policies.h"
#include "test_iterators.h"
#include "test_macros.h"
#include "runway_sample.h"

EXECUTION_POLICY_SFINAE_TEST(is_sorted_until);

static_assert(sfinae_test_is_sorted_until<int, int*, int*, bool (*)(int, int)>);
static_assert(!sfinae_test_is_sorted_until<std::execution::parallel_policy, int*, int*, bool (*)(int, int)>);

// The type X is provided to test that the is_sorted_until algorithm can be used with custom non-movable/non-copyable types.
struct X {
  X() = delete;
  explicit X(int i) : i_(i) {}
  X(const X&)            = delete;
  X(X&&)                 = delete;
  X& operator=(const X&) = delete;
  X& operator=(X&&)      = delete;
  int value() const { return i_; }

private:
  int i_;
};

struct Comp {
  bool operator()(int lhs, int rhs) const { return lhs > rhs; }
  bool operator()(const X& lhs, const X& rhs) const { return lhs.value() > rhs.value(); }
};

template <class Iter>
struct Test {
  template <class ExecutionPolicy>
  void operator()(ExecutionPolicy&& policy) {
    { // check the return types
      int a[]  = {0};
      auto res = std::is_sorted_until(policy, Iter(std::begin(a)), Iter(std::begin(a)), Comp{});
      static_assert(std::is_same_v<decltype(res), Iter>);
    }
    { // empty range
      int a[]  = {0};
      auto res = std::is_sorted_until(policy, Iter(std::begin(a)), Iter(std::begin(a)), Comp{});
      assert(res == Iter(std::begin(a)));
    }
    { // single element range
      int a[]  = {0};
      auto res = std::is_sorted_until(policy, Iter(std::begin(a)), Iter(std::end(a)), Comp{});
      assert(res == Iter(std::end(a)));
    }
    { // two element range, equal - sorted
      int a[] = {0, 0};
      assert(std::is_sorted_until(policy, Iter(std::begin(a)), Iter(std::end(a)), Comp{}) == Iter(std::end(a)));
    }
    { // two element range, ascending - unsorted
      int a[] = {0, 1};
      assert(std::is_sorted_until(policy, Iter(std::begin(a)), Iter(std::end(a)), Comp{}) == Iter(std::begin(a) + 1));
    }
    { // two element range, descending - sorted
      int a[] = {1, 0};
      assert(std::is_sorted_until(policy, Iter(std::begin(a)), Iter(std::end(a)), Comp{}) == Iter(std::end(a)));
    }
    { // two element range, equal - sorted
      int a[] = {1, 1};
      assert(std::is_sorted_until(policy, Iter(std::begin(a)), Iter(std::end(a)), Comp{}) == Iter(std::end(a)));
    }
    { // three element range, equal - sorted
      int a[] = {0, 0, 0};
      assert(std::is_sorted_until(policy, Iter(std::begin(a)), Iter(std::end(a)), Comp{}) == Iter(std::end(a)));
    }
    { // three element range, ascending - unsorted
      int a[] = {0, 0, 1};
      assert(std::is_sorted_until(policy, Iter(std::begin(a)), Iter(std::end(a)), Comp{}) == Iter(std::begin(a) + 2));
    }
    { // three element range, not sorted
      int a[] = {0, 1, 0};
      assert(std::is_sorted_until(policy, Iter(std::begin(a)), Iter(std::end(a)), Comp{}) == Iter(std::begin(a) + 1));
    }
    { // three element range, ascending - unsorted
      int a[] = {0, 1, 1};
      assert(std::is_sorted_until(policy, Iter(std::begin(a)), Iter(std::end(a)), Comp{}) == Iter(std::begin(a) + 1));
    }
    { // three element range, descending -  sorted
      int a[] = {1, 0, 0};
      assert(std::is_sorted_until(policy, Iter(std::begin(a)), Iter(std::end(a)), Comp{}) == Iter(std::end(a)));
    }
    { // three element range, not sorted
      int a[] = {1, 0, 1};
      assert(std::is_sorted_until(policy, Iter(std::begin(a)), Iter(std::end(a)), Comp{}) == Iter(std::begin(a) + 2));
    }
    { // three element range, sorted
      int a[] = {1, 1, 0};
      assert(std::is_sorted_until(policy, Iter(std::begin(a)), Iter(std::end(a)), Comp{}) == Iter(std::end(a)));
    }
    { // three element range, equal - sorted
      int a[] = {1, 1, 1};
      assert(std::is_sorted_until(policy, Iter(std::begin(a)), Iter(std::end(a)), Comp{}) == Iter(std::end(a)));
    }
    { // four element range, equal - sorted
      int a[] = {0, 0, 0, 0};
      assert(std::is_sorted_until(policy, Iter(std::begin(a)), Iter(std::end(a)), Comp{}) == Iter(std::end(a)));
    }
    { // four element range, ascending - unsorted
      int a[] = {0, 0, 0, 1};
      assert(std::is_sorted_until(policy, Iter(std::begin(a)), Iter(std::end(a)), Comp{}) == Iter(std::begin(a) + 3));
    }
    { // four element range, not sorted
      int a[] = {0, 0, 1, 0};
      assert(std::is_sorted_until(policy, Iter(std::begin(a)), Iter(std::end(a)), Comp{}) == Iter(std::begin(a) + 2));
    }
    { // four element range, ascending - unsorted
      int a[] = {0, 0, 1, 1};
      assert(std::is_sorted_until(policy, Iter(std::begin(a)), Iter(std::end(a)), Comp{}) == Iter(std::begin(a) + 2));
    }
    { // four element range, not sorted
      int a[] = {0, 1, 0, 0};
      assert(std::is_sorted_until(policy, Iter(std::begin(a)), Iter(std::end(a)), Comp{}) == Iter(std::begin(a) + 1));
    }
    { // four element range, not sorted
      int a[] = {0, 1, 0, 1};
      assert(std::is_sorted_until(policy, Iter(std::begin(a)), Iter(std::end(a)), Comp{}) == Iter(std::begin(a) + 1));
    }
    { // four element range, not sorted
      int a[] = {0, 1, 1, 0};
      assert(std::is_sorted_until(policy, Iter(std::begin(a)), Iter(std::end(a)), Comp{}) == Iter(std::begin(a) + 1));
    }
    { // four element range, ascending - unsorted
      int a[] = {0, 1, 1, 1};
      assert(std::is_sorted_until(policy, Iter(std::begin(a)), Iter(std::end(a)), Comp{}) == Iter(std::begin(a) + 1));
    }
    { // four element range, descending - sorted
      int a[] = {1, 0, 0, 0};
      assert(std::is_sorted_until(policy, Iter(std::begin(a)), Iter(std::end(a)), Comp{}) == Iter(std::end(a)));
    }
    { // four element range, not sorted
      int a[] = {1, 0, 0, 1};
      assert(std::is_sorted_until(policy, Iter(std::begin(a)), Iter(std::end(a)), Comp{}) == Iter(std::begin(a) + 3));
    }
    { // four element range, not sorted
      int a[] = {1, 0, 1, 0};
      assert(std::is_sorted_until(policy, Iter(std::begin(a)), Iter(std::end(a)), Comp{}) == Iter(std::begin(a) + 2));
    }
    { // four element range, not sorted
      int a[] = {1, 0, 1, 1};
      assert(std::is_sorted_until(policy, Iter(std::begin(a)), Iter(std::end(a)), Comp{}) == Iter(std::begin(a) + 2));
    }
    { // four element range, descending - sorted
      int a[] = {1, 1, 0, 0};
      assert(std::is_sorted_until(policy, Iter(std::begin(a)), Iter(std::end(a)), Comp{}) == Iter(std::end(a)));
    }
    { // four element range, not sorted
      int a[] = {1, 1, 0, 1};
      assert(std::is_sorted_until(policy, Iter(std::begin(a)), Iter(std::end(a)), Comp{}) == Iter(std::begin(a) + 3));
    }
    { // four element range, descending - sorted
      int a[] = {1, 1, 1, 0};
      assert(std::is_sorted_until(policy, Iter(std::begin(a)), Iter(std::end(a)), Comp{}) == Iter(std::end(a)));
    }
    { // four element range, equal - sorted
      int a[] = {1, 1, 1, 1};
      assert(std::is_sorted_until(policy, Iter(std::begin(a)), Iter(std::end(a)), Comp{}) == Iter(std::end(a)));
    }
    { // four element range, min/max - unsorted
      int a[] = {std::numeric_limits<int>::min(),
                 std::numeric_limits<int>::min(),
                 std::numeric_limits<int>::max(),
                 std::numeric_limits<int>::max()};
      assert(std::is_sorted_until(policy, Iter(std::begin(a)), Iter(std::end(a)), Comp{}) == Iter(std::begin(a) + 2));
    }
    { // four element range, min/max - not sorted
      int a[] = {std::numeric_limits<int>::min(),
                 std::numeric_limits<int>::max(),
                 std::numeric_limits<int>::min(),
                 std::numeric_limits<int>::max()};
      assert(std::is_sorted_until(policy, Iter(std::begin(a)), Iter(std::end(a)), Comp{}) == Iter(std::begin(a) + 1));
    }
    { // seven element range, min/max - sorted
      int a[] = {
          std::numeric_limits<int>::min(),
          std::numeric_limits<int>::min() / 2,
          -1,
          0,
          1,
          std::numeric_limits<int>::max() / 2,
          std::numeric_limits<int>::max()};
      assert(std::is_sorted_until(policy, Iter(std::begin(a)), Iter(std::end(a)), Comp{}) == Iter(std::begin(a) + 1));
    }
    { // seven element range, min/max - not sorted
      int a[] = {
          std::numeric_limits<int>::min(),
          std::numeric_limits<int>::min() / 2,
          1,
          0,
          -1,
          std::numeric_limits<int>::max() / 2,
          std::numeric_limits<int>::max()};
      assert(std::is_sorted_until(policy, Iter(std::begin(a)), Iter(std::end(a)), Comp{}) == Iter(std::begin(a) + 1));
    }
    { // many elements, ordering breaks at various sampled positions
      int a[1073];
      std::generate(std::begin(a), std::end(a), [n = 0]() mutable { return n--; });
      runway_sample(std::size(a), [&](size_t i) {
        if (i == 0) {
          return; // sorted range cannot be broken at the first element
        }
        a[i] = 1;
        assert(std::is_sorted_until(policy, Iter(std::begin(a)), Iter(std::end(a)), Comp{}) == Iter(std::begin(a) + i));
        a[i] = -static_cast<int>(i);
      });
    }
  }
};

template <class Iter>
struct TestX {
  template <class ExecutionPolicy>
  void operator()(ExecutionPolicy&& policy) {
    // Four elements, unsorted
    X a[] = {X(2), X(2), X(1), X(3)};
    assert(std::is_sorted_until(policy, Iter(std::begin(a)), Iter(std::end(a)), Comp{}) == Iter(std::begin(a) + 3));
    assert(std::is_sorted_until(policy, Iter(std::begin(a) + 1), Iter(std::end(a)), Comp{}) == Iter(std::begin(a) + 3));
    assert(std::is_sorted_until(policy, Iter(std::begin(a)), Iter(std::begin(a) + 3), Comp{}) ==
           Iter(std::begin(a) + 3));
  }
};

int main(int, char**) {
  types::for_each(types::forward_iterator_list<int*>{}, TestIteratorWithPolicies<Test>{});
  types::for_each(types::forward_iterator_list<const X*>{}, TestIteratorWithPolicies<TestX>{});
  return 0;
}