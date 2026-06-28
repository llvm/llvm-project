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
//           class ForwardIterator1,
//           class ForwardIterator2,
//           class BinaryPredicate>
//   ForwardIterator1 find_first_of(ExecutionPolicy&& exec,
//                                  ForwardIterator1 first1,
//                                  ForwardIterator1 last1,
//                                  ForwardIterator2 first2,
//                                  ForwardIterator2 last2,
//                                  BinaryPredicate p);

#include <algorithm>
#include <cassert>
#include <functional>
#include <limits>
#include <numeric>

#include "test_execution_policies.h"
#include "test_iterators.h"
#include "test_macros.h"
#include "type_algorithms.h"

struct Pred {
  bool operator()(int l, int r) const {
    return l + 1 == r; // ensures that the predicate is not equivalent to std::equal_to
  }
};

template <class Iter>
struct Test {
  template <class ExecutionPolicy>
  void operator()(ExecutionPolicy&& policy) {
    {
      int a[]     = {0};
      unsigned sa = sizeof(a) / sizeof(a[0]);
      assert(std::find_first_of(policy, Iter(a), Iter(a), Iter(a), Iter(a), Pred{}) == Iter(a));
      assert(std::find_first_of(policy, Iter(a), Iter(a + sa), Iter(a), Iter(a), Pred{}) == Iter(a + sa));
    }
    {
      int ia[]    = {0, 1, 2, 3, 0, 1, 2, 3};
      unsigned sa = sizeof(ia) / sizeof(ia[0]);
      int ib[]    = {2, 4, 6, 8};
      unsigned sb = sizeof(ib) / sizeof(ib[0]);
      assert(std::find_first_of(policy, Iter(ia), Iter(ia + sa), Iter(ib), Iter(ib + sb), Pred{}) == Iter(ia + 1));
      int ic[] = {7};
      assert(std::find_first_of(policy, Iter(ia), Iter(ia + sa), Iter(ic), Iter(ic + 1), Pred{}) == Iter(ia + sa));
      assert(std::find_first_of(policy, Iter(ia), Iter(ia + sa), Iter(ic), Iter(ic), Pred{}) == Iter(ia + sa));
      assert(std::find_first_of(policy, Iter(ia), Iter(ia), Iter(ic), Iter(ic + 1), Pred{}) == Iter(ia));
    }
    {
      int a[8192];
      unsigned sa = sizeof(a) / sizeof(a[0]);
      std::iota(a, a + sa, 1);
      a[1023]     = -2;
      a[2048]     = -2;
      a[3071]     = -2;
      int b[]     = {-1, 999999};
      unsigned sb = sizeof(b) / sizeof(b[0]);
      assert(std::find_first_of(policy, Iter(a), Iter(a + sa), Iter(b), Iter(b + sb), Pred{}) == Iter(a + 1023));
    }
    {
      int a[1073];
      unsigned sa = sizeof(a) / sizeof(a[0]);
      std::iota(a, a + sa, 0);
      int b[]     = {1070, 1071, 1072, -1};
      unsigned sb = sizeof(b) / sizeof(b[0]);
      for (unsigned i = 0; i < sa; i = i <= 16 ? i + 1 : unsigned(3.1415 * i)) {
        a[i] = -2;
        assert(std::find_first_of(policy, Iter(a), Iter(a + sa), Iter(b), Iter(b + sb), Pred{}) == Iter(a + i));
        a[i] = i;
      }
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
