//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++17

// UNSUPPORTED: libcpp-has-no-incomplete-pstl

// template<class ExecutionPolicy, class ForwardIterator, class BinaryPredicate>
// ForwardIterator adjacent_find(ExecutionPolicy&& exec,
//                               ForwardIterator first, ForwardIterator last,
//                               BinaryPredicate pred);

#include <algorithm>
#include <cassert>
#include <numeric>
#include <iterator>
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
  int get_value() const { return i_; }

private:
  int i_;
};

struct Pred {
  bool operator()(const X& a, const X& b) const {
    return static_cast<long long>(b.get_value()) - static_cast<long long>(a.get_value()) == 42;
  }
  bool operator()(int a, int b) const { return static_cast<long long>(b) - static_cast<long long>(a) == 42; }
};

template <class Iter>
struct Test {
  template <class ExecutionPolicy>
  void operator()(ExecutionPolicy&& policy) {
    {
      int a[]  = {42};
      auto res = std::adjacent_find(policy, Iter(std::begin(a)), Iter(std::begin(a)), Pred{});
      static_assert(std::is_same_v<decltype(res), Iter>);
    }
    {
      int a[] = {42};
      assert(std::adjacent_find(policy, Iter(std::begin(a)), Iter(std::begin(a)), Pred{}) == Iter(std::begin(a)));
      assert(std::adjacent_find(policy, Iter(std::begin(a)), Iter(std::end(a)), Pred{}) == Iter(std::end(a)));
    }
    {
      int a[] = {42, 84};
      assert(std::adjacent_find(policy, Iter(std::begin(a)), Iter(std::end(a)), Pred{}) == Iter(std::begin(a)));
      assert(std::adjacent_find(policy, Iter(std::begin(a)), Iter(std::begin(a) + 1), Pred{}) ==
             Iter(std::begin(a) + 1));
      assert(std::adjacent_find(policy, Iter(std::begin(a) + 1), Iter(std::end(a)), Pred{}) == Iter(std::end(a)));
    }
    {
      int a[] = {0, 42, 84};
      assert(std::adjacent_find(policy, Iter(std::begin(a)), Iter(std::end(a)), Pred{}) == Iter(std::begin(a)));
    }
    {
      int a[] = {0, 41, 0};
      assert(std::adjacent_find(policy, Iter(std::begin(a)), Iter(std::end(a)), Pred{}) == Iter(std::end(a)));
    }
    {
      int a[] = {0, 1, 0, 42};
      assert(std::adjacent_find(policy, Iter(std::begin(a)), Iter(std::end(a)), Pred{}) == Iter(std::begin(a) + 2));
    }
    {
      int a[] = {0, 41, 0, 41};
      assert(std::adjacent_find(policy, Iter(std::begin(a)), Iter(std::end(a)), Pred{}) == Iter(std::end(a)));
    }
    {
      int a[] = {0, 1, 43, 85};
      assert(std::adjacent_find(policy, Iter(std::begin(a)), Iter(std::end(a)), Pred{}) == Iter(std::begin(a) + 1));
    }
    {
      int a[] = {0, 1, 2, 44, 0, 1, 2, 3};
      assert(std::adjacent_find(policy, Iter(std::begin(a)), Iter(std::end(a)), Pred{}) == Iter(std::begin(a) + 2));
      assert(std::adjacent_find(policy, Iter(std::begin(a) + 2), Iter(std::end(a)), Pred{}) == Iter(std::begin(a) + 2));
      assert(std::adjacent_find(policy, Iter(std::begin(a)), Iter(std::begin(a)), Pred{}) == Iter(std::begin(a)));
      assert(std::adjacent_find(policy, Iter(std::begin(a) + 3), Iter(std::end(a)), Pred{}) == Iter(std::end(a)));
    }
    {
      int a[] = {0, 1, 2, 45, 0, 1, 2, 3};
      assert(std::adjacent_find(policy, Iter(std::begin(a)), Iter(std::end(a)), Pred{}) == Iter(std::end(a)));
    }
    {
      int a[] = {std::numeric_limits<int>::min(),
                 std::numeric_limits<int>::min() + 42,
                 std::numeric_limits<int>::max() - 42,
                 std::numeric_limits<int>::max()};
      assert(std::adjacent_find(policy, Iter(std::begin(a)), Iter(std::end(a)), Pred{}) == Iter(std::begin(a)));
    }
    {
      int a[] = {std::numeric_limits<int>::max(),
                 std::numeric_limits<int>::min(),
                 std::numeric_limits<int>::max() - 42,
                 std::numeric_limits<int>::max()};
      assert(std::adjacent_find(policy, Iter(std::begin(a)), Iter(std::end(a)), Pred{}) == Iter(std::begin(a) + 2));
    }
    {
      int a[] = {std::numeric_limits<int>::max(),
                 std::numeric_limits<int>::min(),
                 std::numeric_limits<int>::max(),
                 std::numeric_limits<int>::min()};
      assert(std::adjacent_find(policy, Iter(std::begin(a)), Iter(std::end(a)), Pred{}) == Iter(std::end(a)));
    }
    {
      int a[] = {std::numeric_limits<int>::max(),
                 std::numeric_limits<int>::min(),
                 std::numeric_limits<int>::min() + 42,
                 std::numeric_limits<int>::max()};
      assert(std::adjacent_find(policy, Iter(std::begin(a)), Iter(std::end(a)), Pred{}) == Iter(std::begin(a) + 1));
    }
    {
      int a[1073];
      std::iota(std::begin(a), std::end(a), 0);
      runway_sample(std::size(a), [&](std::size_t i) {
        if (i == 0)
          return; // skip the first element to avoid out-of-bounds access
        a[i] = a[i - 1] + 42;
        assert(std::adjacent_find(policy, Iter(std::begin(a)), Iter(std::end(a)), Pred{}) ==
               Iter(std::begin(a) + i - 1));
        a[i] = i;
      });
    }
  }
};

template <class Iter>
struct TestX {
  template <class ExecutionPolicy>
  void operator()(ExecutionPolicy&& policy) {
    X a[] = {X(0), X(40), X(82), X(100)};
    assert(std::adjacent_find(policy, Iter(std::begin(a)), Iter(std::end(a)), Pred{}) == Iter(std::begin(a) + 1));
    assert(std::adjacent_find(policy, Iter(std::begin(a) + 1), Iter(std::end(a)), Pred{}) == Iter(std::begin(a) + 1));
    assert(std::adjacent_find(policy, Iter(std::begin(a) + 2), Iter(std::end(a)), Pred{}) == Iter(std::end(a)));
  }
};

int main(int, char**) {
  types::for_each(types::forward_iterator_list<const int*>{}, TestIteratorWithPolicies<Test>{});
  types::for_each(types::forward_iterator_list<const X*>{}, TestIteratorWithPolicies<TestX>{});
  return 0;
}
