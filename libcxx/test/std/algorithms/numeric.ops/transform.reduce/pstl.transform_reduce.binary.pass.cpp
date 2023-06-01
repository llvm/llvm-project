//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// UNSUPPORTED: libcpp-has-no-incomplete-pstl

// <algorithm>

// template<class ExecutionPolicy,
//          class ForwardIterator1, class ForwardIterator2, class T>
//   T transform_reduce(ExecutionPolicy&& exec,
//                      ForwardIterator1 first1, ForwardIterator1 last1,
//                      ForwardIterator2 first2,
//                      T init);
//
// template<class ExecutionPolicy,
//          class ForwardIterator1, class ForwardIterator2, class T,
//          class BinaryOperation1, class BinaryOperation2>
//   T transform_reduce(ExecutionPolicy&& exec,
//                      ForwardIterator1 first1, ForwardIterator1 last1,
//                      ForwardIterator2 first2,
//                      T init,
//                      BinaryOperation1 binary_op1,
//                      BinaryOperation2 binary_op2);

#include <numeric>
#include <vector>

#include "MoveOnly.h"
#include "test_execution_policies.h"
#include "test_iterators.h"
#include "test_macros.h"
#include "type_algorithms.h"

template <class Iter1, class Iter2, class ValueT>
struct Test {
  template <class Policy>
  void operator()(Policy&& policy) {
    for (const auto& pair : {std::pair{0, 34}, {1, 33}, {2, 30}, {100, 313434}, {350, 14046934}}) {
      auto [size, expected] = pair;
      std::vector<int> a(size);
      std::vector<int> b(size);
      for (int i = 0; i != size; ++i) {
        a[i] = i + 1;
        b[i] = i - 4;
      }

      decltype(auto) ret = std::transform_reduce(
          policy,
          Iter1(std::data(a)),
          Iter1(std::data(a) + std::size(a)),
          Iter2(std::data(b)),
          ValueT(34),
          [](ValueT i, ValueT j) { return i + j + 3; },
          [](ValueT i, ValueT j) { return i * j; });
      static_assert(std::is_same_v<decltype(ret), ValueT>);
      assert(ret == expected);
    }

    for (const auto& pair : {std::pair{0, 34}, {1, 30}, {2, 24}, {100, 313134}, {350, 14045884}}) {
      auto [size, expected] = pair;
      std::vector<int> a(size);
      std::vector<int> b(size);
      for (int i = 0; i != size; ++i) {
        a[i] = i + 1;
        b[i] = i - 4;
      }

      decltype(auto) ret = std::transform_reduce(
          policy, Iter1(std::data(a)), Iter1(std::data(a) + std::size(a)), Iter2(std::data(b)), 34);
      static_assert(std::is_same_v<decltype(ret), int>);
      assert(ret == expected);
    }
  }
};

int main(int, char**) {
  types::for_each(
      types::forward_iterator_list<int*>{}, types::apply_type_identity{[](auto v) {
        using Iter2 = typename decltype(v)::type;
        types::for_each(
            types::forward_iterator_list<int*>{}, types::apply_type_identity{[](auto v2) {
              using Iter1 = typename decltype(v2)::type;
              types::for_each(
                  types::type_list<int, MoveOnly>{},
                  TestIteratorWithPolicies<types::partial_instantiation<Test, Iter1, Iter2>::template apply>{});
            }});
      }});

  return 0;
}
