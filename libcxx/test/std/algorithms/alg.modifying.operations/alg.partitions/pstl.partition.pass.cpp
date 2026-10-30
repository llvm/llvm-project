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

// template <class ExecutionPolicy,
//           class ForwardIterator,
//           class Predicate>
//   ForwardIterator partition(ExecutionPolicy&& exec,
//                             ForwardIterator first,
//                             ForwardIterator last,
//                             Predicate pred);

#include <algorithm>
#include <cassert>
#include <functional>
#include <iterator>
#include <limits>
#include <numeric>
#include <type_traits>

#include "test_execution_policies.h"
#include "test_iterators.h"
#include "test_macros.h"
#include "type_algorithms.h"

EXECUTION_POLICY_SFINAE_TEST(partition);

static_assert(sfinae_test_partition<int, int*, int*, bool (*)(int, int)>);
static_assert(!sfinae_test_partition<std::execution::parallel_policy, int*, int*, bool (*)(int, int)>);

template <class Iter>
struct Test {
  template <class ExecutionPolicy>
  void operator()(ExecutionPolicy&& policy) {
    auto is_odd = [](int i) { return i % 2 == 1; };
    { // Check the return type
      int a[] = {0};
      auto r  = std::partition(policy, Iter(std::begin(a)), Iter(std::end(a)), is_odd);
      static_assert(std::is_same_v<decltype(r), Iter>);
    }
    { // Check mixed
      int a[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
      auto r  = std::partition(Iter(std::begin(a)), Iter(std::end(a)), is_odd);
      assert(r == Iter(std::begin(a) + 5));
      for (auto i = Iter(std::begin(a)); i != r; ++i)
        assert(is_odd(*i));
      for (auto i = r; i != Iter(std::end(a)); ++i)
        assert(!is_odd(*i));
    }
    { // Check empty
      int a[] = {0};
      auto r  = std::partition(Iter(std::begin(a)), Iter(std::begin(a)), is_odd);
      assert(r == Iter(std::begin(a)));
    }
    { // Check all false
      int a[10] = {0};
      std::generate(std::begin(a), std::end(a), [v = 0]() mutable { return 2 * v++; });
      auto r = std::partition(Iter(std::begin(a)), Iter(std::end(a)), is_odd);
      assert(r == Iter(std::begin(a)));
    }
    { // Check all true
      int a[10] = {0};
      std::generate(std::begin(a), std::end(a), [v = 0]() mutable { return 1 + 2 * v++; });
      auto r = std::partition(Iter(std::begin(a)), Iter(std::end(a)), is_odd);
      assert(r == Iter(std::end(a)));
    }
    { // Check all true but last
      int a[10] = {0};
      std::generate(std::begin(a), std::end(a), [v = 0]() mutable { return 1 + 2 * v++; });
      a[std::size(a) - 1] = 10;
      auto r              = std::partition(Iter(std::begin(a)), Iter(std::end(a)), is_odd);
      assert(r == Iter(std::end(a) - 1));
      for (auto i = Iter(std::begin(a)); i != r; ++i)
        assert(is_odd(*i));
      for (auto i = r; i != Iter(std::end(a)); ++i)
        assert(!is_odd(*i));
    }
    { // Check all true but first
      int a[10] = {0};
      std::generate(std::begin(a), std::end(a), [v = 0]() mutable { return 1 + 2 * v++; });
      a[0]   = 10;
      auto r = std::partition(Iter(std::begin(a)), Iter(std::end(a)), is_odd);
      assert(r == Iter(std::end(a) - 1));
      for (auto i = Iter(std::begin(a)); i != r; ++i)
        assert(is_odd(*i));
      for (auto i = r; i != Iter(std::end(a)); ++i)
        assert(!is_odd(*i));
    }
    { // Check all false but last
      int a[10] = {0};
      std::generate(std::begin(a), std::end(a), [v = 0]() mutable { return 2 * v++; });
      a[std::size(a) - 1] = 11;
      auto r              = std::partition(Iter(std::begin(a)), Iter(std::end(a)), is_odd);
      assert(r == Iter(std::begin(a) + 1));
      for (auto i = Iter(std::begin(a)); i != r; ++i)
        assert(is_odd(*i));
      for (auto i = r; i != Iter(std::end(a)); ++i)
        assert(!is_odd(*i));
    }
    { // Check all false but first
      int a[10] = {0};
      std::generate(std::begin(a), std::end(a), [v = 0]() mutable { return 2 * v++; });
      a[0]   = 11;
      auto r = std::partition(Iter(std::begin(a)), Iter(std::end(a)), is_odd);
      assert(r == Iter(std::begin(a) + 1));
      for (auto i = Iter(std::begin(a)); i != r; ++i)
        assert(is_odd(*i));
      for (auto i = r; i != Iter(std::end(a)); ++i)
        assert(!is_odd(*i));
    }
  }
};

int main(int, char**) {
  types::for_each(types::forward_iterator_list<int*>{}, TestIteratorWithPolicies<Test>{});
  return 0;
}
