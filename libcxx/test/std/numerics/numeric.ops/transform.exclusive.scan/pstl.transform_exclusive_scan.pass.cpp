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
  void operator()(const Policy& policy) const {
    for (const int size : {0, 1, 2, 100, 350, 10'000}) {
      std::vector<int> data(size);
      std::iota(data.begin(), data.end(), 7);

      const int init = 42;
      int* first     = data.data();
      int* last      = first + size;

      { // general smoke test
        std::vector<int> expected(size);
        std::transform_exclusive_scan(first, last, expected.begin(), init, std::plus{}, [](int x) { return x + 1; });

        std::vector<int> result(size);
        auto ret = std::transform_exclusive_scan(
            policy,
            Iter(first),
            Iter(last),
            result.data(),
            init,
            [check = std::string("Banane")](int i, int j) {
              assert(check == "Banane");
              return i + j;
            },
            [check = std::string("Banane")](int i) {
              assert(check == "Banane");
              return i + 1;
            });
        static_assert(std::is_same_v<decltype(ret), int*>);
        assert(ret == result.data() + size);
        assert(result == expected);
      }

      { // binary_op whose identity element is not int{}
        const auto binary_op = std::multiplies{};
        const auto unary_op  = [](int x) { return (x % 1'000 == 0) ? 2 : 1; }; // keeps product below 2^10

        std::vector<int> expected(size);
        std::transform_exclusive_scan(first, last, expected.begin(), init, binary_op, unary_op);

        std::vector<int> result(size);
        auto ret =
            std::transform_exclusive_scan(policy, Iter(first), Iter(last), result.data(), init, binary_op, unary_op);
        static_assert(std::is_same_v<decltype(ret), int*>);
        assert(ret == result.data() + size);
        assert(result == expected);
      }
    }
  }
};

int main(int, char**) {
  types::for_each(types::forward_iterator_list<int*>{}, TestIteratorWithPolicies<Test>{});
  return 0;
}
