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
//           class BidirectionalIterator,
//           class ForwardIterator>
//   ForwardIterator reverse_copy(ExecutionPolicy&& exec,
//                                BidirectionalIterator first,
//                                BidirectionalIterator last,
//                                ForwardIterator result);

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

EXECUTION_POLICY_SFINAE_TEST(reverse_copy);

static_assert(sfinae_test_reverse_copy<int, int*, int*, int*>);
static_assert(!sfinae_test_reverse_copy<std::execution::parallel_policy, int*, int*, int*>);

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
      int ia[]              = {0};
      int ib[std::size(ia)] = {-1};
      Iter2 r = std::reverse_copy(policy, Iter1(std::begin(ia)), Iter1(std::begin(ia)), Iter2(std::begin(ib)));
      assert(r == Iter2(std::begin(ib)));
      assert(ib[0] == -1);
      r = std::reverse_copy(policy, Iter1(std::begin(ia)), Iter1(std::end(ia)), Iter2(std::begin(ib)));
      assert(ia[0] == 0);
    }
    {
      int ia[]              = {0, 1};
      int ib[std::size(ia)] = {-1};
      Iter2 r = std::reverse_copy(policy, Iter1(std::begin(ia)), Iter1(std::end(ia)), Iter2(std::begin(ib)));
      assert(r == Iter2(std::end(ib)));
      assert(ib[0] == 1);
      assert(ib[1] == 0);
    }
    {
      int ia[]              = {0, 1, 2};
      int ib[std::size(ia)] = {-1};
      Iter2 r = std::reverse_copy(policy, Iter1(std::begin(ia)), Iter1(std::end(ia)), Iter2(std::begin(ib)));
      assert(r == Iter2(std::end(ib)));
      assert(ib[0] == 2);
      assert(ib[1] == 1);
      assert(ib[2] == 0);
    }
    {
      int ia[]              = {0, 1, 2, 3};
      int ib[std::size(ia)] = {-1};
      Iter2 r = std::reverse_copy(policy, Iter1(std::begin(ia)), Iter1(std::end(ia)), Iter2(std::begin(ib)));
      assert(r == Iter2(std::end(ib)));
      assert(ib[0] == 3);
      assert(ib[1] == 2);
      assert(ib[2] == 1);
      assert(ib[3] == 0);
    }
    {
      int ia[1073];
      int ib[1073];
      std::iota(std::begin(ia), std::end(ia), 1);
      runway_sample(std::size(ia) + 1, [&](size_t i) {
        Iter2 r = std::reverse_copy(policy, Iter1(std::begin(ia)), Iter1(std::begin(ia) + i), Iter2(std::begin(ib)));
        assert(r == Iter2(std::begin(ib) + i));
        for (size_t j = 0; j < i; ++j) {
          assert(ib[j] == static_cast<int>(i - j));
        }
      });
    }
  }
};

int main(int, char**) {
  types::for_each(
      types::concatenate_t<types::bidirectional_iterator_list<int*>, types::bidirectional_iterator_list<const int*>>{},
      types::apply_type_identity{[](auto v) {
        using Iter = typename decltype(v)::type;
        types::for_each(types::forward_iterator_list<int*>{},
                        TestIteratorWithPolicies<types::partial_instantiation<Test, Iter>::template apply>{});
      }});
}
