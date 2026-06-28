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
//           class ForwardIterator2>
//   ForwardIterator1 find_first_of(ExecutionPolicy&& exec,
//                                  ForwardIterator1 first1,
//                                  ForwardIterator1 last1,
//                                  ForwardIterator2 first2,
//                                  ForwardIterator2 last2);

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
      assert(std::find_first_of(policy, Iter(a), Iter(a), Iter(a), Iter(a)) == Iter(a));
      assert(std::find_first_of(policy, Iter(a), Iter(a + sa), Iter(a), Iter(a)) == Iter(a + sa));
    }
    {
      int ia[]    = {0, 1, 2, 3, 0, 1, 2, 3};
      unsigned sa = sizeof(ia) / sizeof(ia[0]);
      int ib[]    = {1, 3, 5, 7};
      unsigned sb = sizeof(ib) / sizeof(ib[0]);
      assert(std::find_first_of(Iter(ia), Iter(ia + sa), Iter(ib), Iter(ib + sb)) == Iter(ia + 1));
      int ic[] = {7};
      assert(std::find_first_of(Iter(ia), Iter(ia + sa), Iter(ic), Iter(ic + 1)) == Iter(ia + sa));
      assert(std::find_first_of(Iter(ia), Iter(ia + sa), Iter(ic), Iter(ic)) == Iter(ia + sa));
      assert(std::find_first_of(Iter(ia), Iter(ia), Iter(ic), Iter(ic + 1)) == Iter(ia));
    }
    {
      int a[8192];
      unsigned sa = sizeof(a) / sizeof(a[0]);
      std::iota(a, a + sa, 1);
      a[1023]     = -1;
      a[2048]     = -1;
      a[3071]     = -1;
      int b[]     = {-1, 999999};
      unsigned sb = sizeof(b) / sizeof(b[0]);
      assert(std::find_first_of(policy, Iter(a), Iter(a + sa), Iter(b), Iter(b + sb)) == Iter(a + 1023));
    }
    {
      int a[1073];
      unsigned sa = sizeof(a) / sizeof(a[0]);
      std::iota(a, a + sa, 0);
      int b[]     = {1070, 1071, 1072, -1};
      unsigned sb = sizeof(b) / sizeof(b[0]);
      for (unsigned i = 0; i < sa; i = i <= 16 ? i + 1 : unsigned(3.1415 * i)) {
        a[i] = -1;
        assert(std::find_first_of(policy, Iter(a), Iter(a + sa), Iter(b), Iter(b + sb)) == Iter(a + i));
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
