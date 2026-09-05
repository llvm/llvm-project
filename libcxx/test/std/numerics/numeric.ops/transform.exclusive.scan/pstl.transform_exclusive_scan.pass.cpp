//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++17

// UNSUPPORTED: libcpp-has-no-incomplete-pstl

// <numeric>

// template<class ExecutionPolicy,
//          class ForwardIterator1, class ForwardIterator2, class T,
//          class BinaryOperation, class UnaryOperation>
//   ForwardIterator2
//   transform_exclusive_scan(ExecutionPolicy&& exec,
//                            ForwardIterator1 first, ForwardIterator1 last,
//                            ForwardIterator2 result,
//                            T init,
//                            BinaryOperation binary_op,
//                            UnaryOperation unary_op);

#include <cassert>
#include <functional>
#include <numeric>
#include <string>
#include <vector>

#include "test_execution_policies.h"
#include "test_iterators.h"
#include "type_algorithms.h"

template <class Iter>
struct Test {
  template <class Policy>
  void operator()(Policy&& policy) {
    for (int size : {0, 1, 2, 100, 350, 10000}) {
      std::vector<int> a(size);
      for (int i = 0; i != size; ++i)
        a[i] = i;

      std::vector<int> expected(size);
      std::transform_exclusive_scan(a.begin(), a.end(), expected.begin(), 0, std::plus{}, [](int x) { return x + 1; });

      std::vector<int> result(size);
      auto ret = std::transform_exclusive_scan(
          policy,
          Iter(std::data(a)),
          Iter(std::data(a) + std::size(a)),
          std::data(result),
          0,
          [check = std::string("Banane")](int i, int j) {
            assert(check == "Banane");
            return i + j;
          },
          [check = std::string("Banane")](int i) {
            assert(check == "Banane");
            return i + 1;
          });
      static_assert(std::is_same_v<decltype(ret), int*>);
      assert(ret == std::data(result) + size);
      assert(result == expected);
    }
  }
};

int main(int, char**) {
  types::for_each(types::forward_iterator_list<int*>{}, TestIteratorWithPolicies<Test>{});
  return 0;
}
