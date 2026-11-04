//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// UNSUPPORTED: libcpp-has-no-incomplete-pstl

// template<class ExecutionPolicy, class ForwardIterator, class Comp>
//   bool is_sorted(ExecutionPolicy&& exec,
//                  ForwardIterator first, ForwardIterator last,
//                  Comp comp);

#include <algorithm>
#include <cassert>
#include <functional>
#include <limits>
#include <numeric>

#include "test_execution_policies.h"
#include "test_iterators.h"
#include "test_macros.h"
#include "type_algorithms.h"

template <class Iter>
struct Test {
  template <class ExecutionPolicy>
  void operator()(ExecutionPolicy&& policy) {
    {
      int a[]     = {0};
      unsigned sa = sizeof(a) / sizeof(a[0]);
      assert(std::is_sorted(policy, Iter(a), Iter(a)));
      assert(std::is_sorted(policy, Iter(a), Iter(a + sa), std::greater<int>()));
    }
    {
      int a[]     = {0, 0};
      unsigned sa = sizeof(a) / sizeof(a[0]);
      assert(std::is_sorted(policy, Iter(a), Iter(a + sa), std::greater<int>()));
    }
    {
      int a[]     = {0, 1};
      unsigned sa = sizeof(a) / sizeof(a[0]);
      assert(!std::is_sorted(policy, Iter(a), Iter(a + sa), std::greater<int>()));
    }
    {
      int a[]     = {1, 0};
      unsigned sa = sizeof(a) / sizeof(a[0]);
      assert(std::is_sorted(policy, Iter(a), Iter(a + sa), std::greater<int>()));
    }
    {
      int a[]     = {1, 1};
      unsigned sa = sizeof(a) / sizeof(a[0]);
      assert(std::is_sorted(policy, Iter(a), Iter(a + sa), std::greater<int>()));
    }

    {
      int a[]     = {0, 0, 0};
      unsigned sa = sizeof(a) / sizeof(a[0]);
      assert(std::is_sorted(policy, Iter(a), Iter(a + sa), std::greater<int>()));
    }
    {
      int a[]     = {0, 0, 1};
      unsigned sa = sizeof(a) / sizeof(a[0]);
      assert(!std::is_sorted(policy, Iter(a), Iter(a + sa), std::greater<int>()));
    }
    {
      int a[]     = {0, 1, 0};
      unsigned sa = sizeof(a) / sizeof(a[0]);
      assert(!std::is_sorted(policy, Iter(a), Iter(a + sa), std::greater<int>()));
    }
    {
      int a[]     = {0, 1, 1};
      unsigned sa = sizeof(a) / sizeof(a[0]);
      assert(!std::is_sorted(policy, Iter(a), Iter(a + sa), std::greater<int>()));
    }
    {
      int a[]     = {1, 0, 0};
      unsigned sa = sizeof(a) / sizeof(a[0]);
      assert(std::is_sorted(policy, Iter(a), Iter(a + sa), std::greater<int>()));
    }
    {
      int a[]     = {1, 0, 1};
      unsigned sa = sizeof(a) / sizeof(a[0]);
      assert(!std::is_sorted(policy, Iter(a), Iter(a + sa), std::greater<int>()));
    }
    {
      int a[]     = {1, 1, 0};
      unsigned sa = sizeof(a) / sizeof(a[0]);
      assert(std::is_sorted(policy, Iter(a), Iter(a + sa), std::greater<int>()));
    }
    {
      int a[]     = {1, 1, 1};
      unsigned sa = sizeof(a) / sizeof(a[0]);
      assert(std::is_sorted(policy, Iter(a), Iter(a + sa), std::greater<int>()));
    }

    {
      int a[]     = {0, 0, 0, 0};
      unsigned sa = sizeof(a) / sizeof(a[0]);
      assert(std::is_sorted(policy, Iter(a), Iter(a + sa), std::greater<int>()));
    }
    {
      int a[]     = {0, 0, 0, 1};
      unsigned sa = sizeof(a) / sizeof(a[0]);
      assert(!std::is_sorted(policy, Iter(a), Iter(a + sa), std::greater<int>()));
    }
    {
      int a[]     = {0, 0, 1, 0};
      unsigned sa = sizeof(a) / sizeof(a[0]);
      assert(!std::is_sorted(policy, Iter(a), Iter(a + sa), std::greater<int>()));
    }
    {
      int a[]     = {0, 0, 1, 1};
      unsigned sa = sizeof(a) / sizeof(a[0]);
      assert(!std::is_sorted(policy, Iter(a), Iter(a + sa), std::greater<int>()));
    }
    {
      int a[]     = {0, 1, 0, 0};
      unsigned sa = sizeof(a) / sizeof(a[0]);
      assert(!std::is_sorted(policy, Iter(a), Iter(a + sa), std::greater<int>()));
    }
    {
      int a[]     = {0, 1, 0, 1};
      unsigned sa = sizeof(a) / sizeof(a[0]);
      assert(!std::is_sorted(policy, Iter(a), Iter(a + sa), std::greater<int>()));
    }
    {
      int a[]     = {0, 1, 1, 0};
      unsigned sa = sizeof(a) / sizeof(a[0]);
      assert(!std::is_sorted(policy, Iter(a), Iter(a + sa), std::greater<int>()));
    }
    {
      int a[]     = {0, 1, 1, 1};
      unsigned sa = sizeof(a) / sizeof(a[0]);
      assert(!std::is_sorted(policy, Iter(a), Iter(a + sa), std::greater<int>()));
    }
    {
      int a[]     = {1, 0, 0, 0};
      unsigned sa = sizeof(a) / sizeof(a[0]);
      assert(std::is_sorted(policy, Iter(a), Iter(a + sa), std::greater<int>()));
    }
    {
      int a[]     = {1, 0, 0, 1};
      unsigned sa = sizeof(a) / sizeof(a[0]);
      assert(!std::is_sorted(policy, Iter(a), Iter(a + sa), std::greater<int>()));
    }
    {
      int a[]     = {1, 0, 1, 0};
      unsigned sa = sizeof(a) / sizeof(a[0]);
      assert(!std::is_sorted(policy, Iter(a), Iter(a + sa), std::greater<int>()));
    }
    {
      int a[]     = {1, 0, 1, 1};
      unsigned sa = sizeof(a) / sizeof(a[0]);
      assert(!std::is_sorted(policy, Iter(a), Iter(a + sa), std::greater<int>()));
    }
    {
      int a[]     = {1, 1, 0, 0};
      unsigned sa = sizeof(a) / sizeof(a[0]);
      assert(std::is_sorted(policy, Iter(a), Iter(a + sa), std::greater<int>()));
    }
    {
      int a[]     = {1, 1, 0, 1};
      unsigned sa = sizeof(a) / sizeof(a[0]);
      assert(!std::is_sorted(policy, Iter(a), Iter(a + sa), std::greater<int>()));
    }
    {
      int a[]     = {1, 1, 1, 0};
      unsigned sa = sizeof(a) / sizeof(a[0]);
      assert(std::is_sorted(policy, Iter(a), Iter(a + sa), std::greater<int>()));
    }
    {
      int a[]     = {1, 1, 1, 1};
      unsigned sa = sizeof(a) / sizeof(a[0]);
      assert(std::is_sorted(policy, Iter(a), Iter(a + sa), std::greater<int>()));
    }
    {
      int a[]     = {std::numeric_limits<int>::max(),
                     std::numeric_limits<int>::max(),
                     std::numeric_limits<int>::min(),
                     std::numeric_limits<int>::min()};
      unsigned sa = sizeof(a) / sizeof(a[0]);
      assert(std::is_sorted(policy, Iter(a), Iter(a + sa), std::greater<int>()));
    }
    {
      int a[]     = {std::numeric_limits<int>::min(),
                     std::numeric_limits<int>::max(),
                     std::numeric_limits<int>::min(),
                     std::numeric_limits<int>::max()};
      unsigned sa = sizeof(a) / sizeof(a[0]);
      assert(!std::is_sorted(policy, Iter(a), Iter(a + sa), std::greater<int>()));
    }
    {
      int a[] = {
          std::numeric_limits<int>::max(),
          std::numeric_limits<int>::max() / 2,
          1,
          0,
          -1,
          std::numeric_limits<int>::min() / 2,
          std::numeric_limits<int>::min()};
      unsigned sa = sizeof(a) / sizeof(a[0]);
      assert(std::is_sorted(policy, Iter(a), Iter(a + sa), std::greater<int>()));
    }
    {
      int a[] = {
          std::numeric_limits<int>::max(),
          std::numeric_limits<int>::max() / 2,
          -1,
          0,
          1,
          std::numeric_limits<int>::min() / 2,
          std::numeric_limits<int>::min()};
      unsigned sa = sizeof(a) / sizeof(a[0]);
      assert(!std::is_sorted(policy, Iter(a), Iter(a + sa), std::greater<int>()));
    }
  }
};

int main(int, char**) {
  types::for_each(types::concatenate_t<types::forward_iterator_list<int*>,
                                       types::bidirectional_iterator_list<int*>,
                                       types::random_access_iterator_list<int*>>{},
                  TestIteratorWithPolicies<Test>{});

  return 0;
}
