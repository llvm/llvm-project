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
#include <iterator>
#include <limits>
#include <numeric>

#include "test_execution_policies.h"
#include "test_iterators.h"
#include "test_macros.h"
#include "type_algorithms.h"

EXECUTION_POLICY_SFINAE_TEST(find_first_of);

static_assert(sfinae_test_find_first_of<int, int*, int*, int*, int*, bool (*)(int, int)>);
static_assert(!sfinae_test_find_first_of<std::execution::parallel_policy, int*, int*, int*, int*, bool (*)(int, int)>);

struct Pred {
  bool operator()(int l, int r) const {
    return l + 1 == r; // ensures that the predicate is not equivalent to std::equal_to
  }
};

template <class Callable>
void runway_sample(size_t size, Callable callable) {
  constexpr size_t affix = 16;
  // 0, 1, 2, ..., 15, 16, 50, 157, 493, 1548, ...
  for (size_t i = 0; i < size; i = i < affix ? i + 1 : size_t(3.1415 * i)) {
    callable(i);
  }
  if (size <= affix)
    return;
  // size - 16, size - 15, ..., size - 1
  for (size_t i = size - affix; i < size; ++i) {
    callable(i);
  }
}

template <class Iter1, class Iter2>
struct Test {
  template <class ExecutionPolicy>
  void operator()(ExecutionPolicy&& policy) {
    {
      int a[] = {0};
      assert(
          std::find_first_of(
              policy, Iter1(std::begin(a)), Iter1(std::begin(a)), Iter2(std::begin(a)), Iter2(std::begin(a)), Pred{}) ==
          Iter1(std::begin(a)));
      assert(
          std::find_first_of(
              policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(a)), Iter2(std::begin(a)), Pred{}) ==
          Iter1(std::end(a)));
    }
    {
      int ia[] = {0, 1, 2, 3, 0, 1, 2, 3};
      int ib[] = {2, 4, 6, 8};
      assert(
          std::find_first_of(
              policy, Iter1(std::begin(ia)), Iter1(std::end(ia)), Iter2(std::begin(ib)), Iter2(std::end(ib)), Pred{}) ==
          Iter1(std::begin(ia) + 1));
      int ic[] = {7};
      assert(
          std::find_first_of(
              policy, Iter1(std::begin(ia)), Iter1(std::end(ia)), Iter2(std::begin(ic)), Iter2(std::end(ic)), Pred{}) ==
          Iter1(std::end(ia)));
      assert(std::find_first_of(
                 policy,
                 Iter1(std::begin(ia)),
                 Iter1(std::end(ia)),
                 Iter2(std::begin(ic)),
                 Iter2(std::begin(ic)),
                 Pred{}) == Iter1(std::end(ia)));
      assert(std::find_first_of(
                 policy,
                 Iter1(std::begin(ia)),
                 Iter1(std::begin(ia)),
                 Iter2(std::begin(ic)),
                 Iter2(std::end(ic)),
                 Pred{}) == Iter1(std::begin(ia)));
    }
    {
      int a[8192];
      std::iota(std::begin(a), std::end(a), 1);
      a[1023] = -2;
      a[2048] = -2;
      a[3071] = -2;
      int b[] = {-1, 999999};
      assert(std::find_first_of(
                 policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(b)), Iter2(std::end(b)), Pred{}) ==
             Iter1(std::begin(a) + 1023));
    }
    {
      int a[1073];
      std::iota(std::begin(a), std::end(a), 0);
      int b[] = {1074, 1075, 1076, -1};
      runway_sample(std::size(a), [&](size_t i) {
        a[i] = -2;
        assert(
            std::find_first_of(
                policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(b)), Iter2(std::end(b)), Pred{}) ==
            Iter1(std::begin(a) + i));
        a[i] = i;
      });
    }
  }
};

int main(int, char**) {
  types::for_each(types::forward_iterator_list<int*>{}, types::apply_type_identity{[](auto v) {
                    using Iter = typename decltype(v)::type;
                    types::for_each(
                        types::forward_iterator_list<int*>{},
                        TestIteratorWithPolicies< types::partial_instantiation<Test, Iter>::template apply>{});
                  }});

  return 0;
}
