//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++17

// UNSUPPORTED: libcpp-has-no-incomplete-pstl

// template<class ExecutionPolicy, class ForwardIterator>
// ForwardIterator adjacent_find(ExecutionPolicy&& exec,
//                               ForwardIterator first, ForwardIterator last);

#include <algorithm>
#include <cassert>
#include <iterator>
#include <numeric>
#include <limits>

#include "test_execution_policies.h"
#include "test_macros.h"
#include "test_iterators.h"
#include "type_algorithms.h"
#include "runway_sample.h"

// The type X is provided to test that the adjacent_find algorithm can be used with custom non-movable/non-copyable types.
struct X {
  X() = delete;
  explicit X(int i) : i_(i) {}
  X(const X&)            = delete;
  X(X&&)                 = delete;
  X& operator=(const X&) = delete;
  X& operator=(X&&)      = delete;
  bool operator==(const X& other) const { return i_ == other.i_; }

private:
  int i_;
};

template <class Iter>
struct Test {
  template <class ExecutionPolicy>
  void operator()(ExecutionPolicy&& policy) {
    { // Check the result type of the algorithm
      int a[]  = {42};
      auto res = std::adjacent_find(policy, Iter(std::begin(a)), Iter(std::begin(a)));
      static_assert(std::is_same_v<decltype(res), Iter>);
    }
    { // Empty range
      int a[] = {42};
      assert(std::adjacent_find(policy, Iter(std::begin(a)), Iter(std::begin(a))) == Iter(std::begin(a)));
    }
    { // Single element range
      int a[] = {42};
      assert(std::adjacent_find(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::end(a)));
    }
    { // Two equal elements
      int a[] = {42, 42};
      assert(std::adjacent_find(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a)));
      assert(std::adjacent_find(policy, Iter(std::begin(a)), Iter(std::begin(a) + 1)) == Iter(std::begin(a) + 1));
      assert(std::adjacent_find(policy, Iter(std::begin(a) + 1), Iter(std::end(a))) == Iter(std::end(a)));
    }
    { // Three elements, all equal
      int a[] = {0, 0, 0};
      assert(std::adjacent_find(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a)));
    }
    { // Three elements, no equal pairs
      int a[] = {0, 1, 0};
      assert(std::adjacent_find(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::end(a)));
    }
    { // Four elements, one equal pair
      int a[] = {0, 1, 0, 0};
      assert(std::adjacent_find(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 2));
    }
    { // Four elements, no equal pairs
      int a[] = {0, 1, 0, 1};
      assert(std::adjacent_find(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::end(a)));
    }
    { // Four elements, two equal pairs
      int a[] = {0, 1, 1, 1};
      assert(std::adjacent_find(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 1));
    }
    { // Eight elements, one equal pair
      int a[] = {0, 1, 2, 2, 0, 1, 2, 3};
      assert(std::adjacent_find(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 2));
      assert(std::adjacent_find(policy, Iter(std::begin(a) + 2), Iter(std::end(a))) == Iter(std::begin(a) + 2));
      assert(std::adjacent_find(policy, Iter(std::begin(a) + 3), Iter(std::end(a))) == Iter(std::end(a)));
    }
    { // Eight elements, no equal pairs
      int a[] = {0, 1, 2, 7, 0, 1, 2, 3};
      assert(std::adjacent_find(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::end(a)));
    }
    { // 1073 iotaed elements, look for equal pair at different sample positions
      int a[1073];
      std::iota(std::begin(a), std::end(a), 0);
      runway_sample(std::size(a), [&](std::size_t i) {
        if (i == 0)
          return; // skip the first element to avoid out-of-bounds access
        a[i] = a[i - 1];
        assert(std::adjacent_find(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + i - 1));
        a[i] = static_cast<int>(i);
      });
    }
  }
};

template <class Iter>
struct TestX {
  template <class ExecutionPolicy>
  void operator()(ExecutionPolicy&& policy) {
    // Four elements, one equal pair
    X a[] = {X(0), X(1), X(1), X(2)};
    assert(std::adjacent_find(policy, Iter(std::begin(a)), Iter(std::end(a))) == Iter(std::begin(a) + 1));
    assert(std::adjacent_find(policy, Iter(std::begin(a) + 1), Iter(std::end(a))) == Iter(std::begin(a) + 1));
    assert(std::adjacent_find(policy, Iter(std::begin(a) + 2), Iter(std::end(a))) == Iter(std::end(a)));
  }
};

int main(int, char**) {
  types::for_each(types::forward_iterator_list<const int*>{}, TestIteratorWithPolicies<Test>{});
  types::for_each(types::forward_iterator_list<const X*>{}, TestIteratorWithPolicies<TestX>{});
  return 0;
}
