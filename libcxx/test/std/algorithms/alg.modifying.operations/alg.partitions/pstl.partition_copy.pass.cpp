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
//           class ForwardOutIterator1,
//           class ForwardOutIterator2,
//           class Predicate>
//   pair<ForwardOutIterator1, ForwardOutIterator2>
//   partition_copy(ExecutionPolicy&& exec,
//                  ForwardIterator first,
//                  ForwardIterator last,
//                  ForwardOutIterator1 out_true,
//                  ForwardOutIterator2 out_false,
//                  Predicate pred);

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

EXECUTION_POLICY_SFINAE_TEST(partition_copy);

static_assert(sfinae_test_partition_copy<int, int*, int*, int*, int*, bool (*)(int)>);
static_assert(!sfinae_test_partition_copy<std::execution::parallel_policy, int*, int*, int*, int*, bool (*)(int)>);

template <class Iter1, class Iter2>
struct Test {
  template <class ExecutionPolicy>
  void operator()(ExecutionPolicy&& policy) {
    auto is_odd = [](int i) { return i % 2 == 1; };
    { // Check the return type
      int a[]             = {0};
      int t[std::size(a)] = {0};
      int f[std::size(a)] = {0};
      auto r              = std::partition_copy(
          policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(t)), Iter2(std::begin(f)), is_odd);
      static_assert(std::is_same_v<decltype(r), std::pair<Iter2, Iter2>>);
    }
    { // Check equal sizes
      int a[]             = {1, 2, 3, 4, 6, 8, 5, 7};
      int t[std::size(a)] = {0};
      int f[std::size(a)] = {0};
      auto r              = std::partition_copy(
          policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(t)), Iter2(std::begin(f)), is_odd);
      assert(r.first == Iter2(std::begin(t) + 4));
      assert(t[0] == 1);
      assert(t[1] == 3);
      assert(t[2] == 5);
      assert(t[3] == 7);
      assert(r.second == Iter2(std::begin(f) + 4));
      assert(f[0] == 2);
      assert(f[1] == 4);
      assert(f[2] == 6);
      assert(f[3] == 8);
    }
    { // Check mixed
      int a[]             = {1, 2, 3, 4, 5, 6, 7, 8, 9};
      int t[std::size(a)] = {0};
      int f[std::size(a)] = {0};
      auto r              = std::partition_copy(
          policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(t)), Iter2(std::begin(f)), is_odd);
      assert(r.first == Iter2(std::begin(t) + 5));
      assert(t[0] == 1);
      assert(t[1] == 3);
      assert(t[2] == 5);
      assert(t[3] == 7);
      assert(t[4] == 9);
      assert(r.second == Iter2(std::begin(f) + 4));
      assert(f[0] == 2);
      assert(f[1] == 4);
      assert(f[2] == 6);
      assert(f[3] == 8);
    }
    { // Check empty
      int a[] = {0};
      int t[] = {-1};
      int f[] = {-1};
      auto r  = std::partition_copy(
          policy, Iter1(std::begin(a)), Iter1(std::begin(a)), Iter2(std::begin(t)), Iter2(std::begin(f)), is_odd);
      assert(r.first == Iter2(std::begin(t)));
      assert(r.second == Iter2(std::begin(f)));
      assert(t[0] == -1);
      assert(f[0] == -1);
    }
    { // Check one true
      int a[] = {1};
      int t[] = {-1};
      int f[] = {-1};
      auto r  = std::partition_copy(
          policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(t)), Iter2(std::begin(f)), is_odd);
      assert(r.first == Iter2(std::end(t)));
      assert(r.second == Iter2(std::begin(f)));
      assert(t[0] == 1);
      assert(f[0] == -1);
    }
    { // Check one false
      int a[] = {0};
      int t[] = {-1};
      int f[] = {-1};
      auto r  = std::partition_copy(
          policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(t)), Iter2(std::begin(f)), is_odd);
      assert(r.first == Iter2(std::begin(t)));
      assert(r.second == Iter2(std::end(f)));
      assert(t[0] == -1);
      assert(f[0] == 0);
    }
    { // Check all false
      int a[10] = {0};
      std::generate(std::begin(a), std::end(a), [v = 0]() mutable { return 2 * v++; });
      int t[std::size(a)] = {0};
      int f[std::size(a)] = {0};
      auto r              = std::partition_copy(
          policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(t)), Iter2(std::begin(f)), is_odd);
      assert(r.first == Iter2(std::begin(t)));
      assert(r.second == Iter2(std::end(f)));
      assert(std::equal(std::begin(a), std::end(a), std::begin(f)));
    }
    { // Check all true
      int a[10] = {0};
      std::generate(std::begin(a), std::end(a), [v = 0]() mutable { return 1 + 2 * v++; });
      int t[std::size(a)] = {0};
      int f[std::size(a)] = {0};
      auto r              = std::partition_copy(
          policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(t)), Iter2(std::begin(f)), is_odd);
      assert(r.first == Iter2(std::end(t)));
      assert(r.second == Iter2(std::begin(f)));
      assert(std::equal(std::begin(a), std::end(a), std::begin(t)));
    }
    { // Check all true but last
      int a[10] = {0};
      std::generate(std::begin(a), std::end(a), [v = 0]() mutable { return 1 + 2 * v++; });
      a[std::size(a) - 1] = 10;
      int t[std::size(a)] = {0};
      int f[std::size(a)] = {0};
      auto r              = std::partition_copy(
          policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(t)), Iter2(std::begin(f)), is_odd);
      assert(r.first == Iter2(std::end(t) - 1));
      assert(r.second == Iter2(std::begin(f) + 1));
      assert(std::equal(std::begin(a), std::end(a) - 1, std::begin(t)));
      assert(f[0] == 10);
    }
    { // Check all true but first
      int a[10] = {0};
      std::generate(std::begin(a), std::end(a), [v = 0]() mutable { return 1 + 2 * v++; });
      a[0]                = 10;
      int t[std::size(a)] = {0};
      int f[std::size(a)] = {0};
      auto r              = std::partition_copy(
          policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(t)), Iter2(std::begin(f)), is_odd);
      assert(r.first == Iter2(std::end(t) - 1));
      assert(r.second == Iter2(std::begin(f) + 1));
      assert(std::equal(std::begin(a) + 1, std::end(a), std::begin(t)));
      assert(f[0] == 10);
    }
    { // Check all false but last
      int a[10] = {0};
      std::generate(std::begin(a), std::end(a), [v = 0]() mutable { return 2 * v++; });
      a[std::size(a) - 1] = 11;
      int t[std::size(a)] = {0};
      int f[std::size(a)] = {0};
      auto r              = std::partition_copy(
          policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(t)), Iter2(std::begin(f)), is_odd);
      assert(r.first == Iter2(std::begin(t) + 1));
      assert(r.second == Iter2(std::end(f) - 1));
      assert(t[0] == 11);
      assert(std::equal(std::begin(a), std::end(a) - 1, std::begin(f)));
    }
    { // Check all false but first
      int a[10] = {0};
      std::generate(std::begin(a), std::end(a), [v = 0]() mutable { return 2 * v++; });
      a[0]                = 11;
      int t[std::size(a)] = {0};
      int f[std::size(a)] = {0};
      auto r              = std::partition_copy(
          policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(t)), Iter2(std::begin(f)), is_odd);
      assert(r.first == Iter2(std::begin(t) + 1));
      assert(r.second == Iter2(std::end(f) - 1));
      assert(t[0] == 11);
      assert(std::equal(std::begin(a) + 1, std::end(a), std::begin(f)));
    }
    { // Check large input
      int a[1000] = {0};
      std::iota(std::begin(a), std::end(a), 0);
      int t[std::size(a)] = {0};
      int f[std::size(a)] = {0};
      auto r              = std::partition_copy(
          policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(t)), Iter2(std::begin(f)), is_odd);
      assert(r.first == Iter2(std::begin(t) + 500));
      assert(r.second == Iter2(std::begin(f) + 500));
      for (int i = 0; i != 500; ++i) {
        assert(t[i] == 2 * i + 1);
        assert(f[i] == 2 * i);
      }
    }
  }
};

int main(int, char**) {
  types::for_each(types::forward_iterator_list<const int*>{}, types::apply_type_identity{[](auto v) {
                    using Iter = typename decltype(v)::type;
                    types::for_each(
                        types::forward_iterator_list<int*>{},
                        TestIteratorWithPolicies<types::partial_instantiation<Test, Iter>::template apply>{});
                  }});
  return 0;
}
