//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// UNSUPPORTED: libcpp-has-no-incomplete-pstl

// <numeric>

// template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2>
//   ForwardIterator2
//     adjacent_difference(ExecutionPolicy&& exec,
//                         ForwardIterator1 first, ForwardIterator1 last, ForwardIterator2 result);
//
// template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2,
//          class BinaryOperation>
//   ForwardIterator2
//     adjacent_difference(ExecutionPolicy&& exec,
//                         ForwardIterator1 first, ForwardIterator1 last,
//                         ForwardIterator2 result, BinaryOperation binary_op);

#include <algorithm>
#include <array>
#include <numeric>
#include <vector>

#include "test_execution_policies.h"
#include "test_iterators.h"

template <class Iter1, class Iter2>
struct Test {
  template <int N, int OutN = N - 1, class Policy>
  void test(Policy&& policy,
            std::array<int, N> input,
            std::array<int, OutN> plus_expected,
            std::array<int, OutN> minus_expected) {
    {
      std::array<int, OutN> out;
      auto ret =
          std::adjacent_difference(policy, Iter1(input.data()), Iter1(input.data() + input.size()), Iter2(out.data()));
      assert(base(ret) == out.data() + out.size());
      assert(out == minus_expected);
    }
    {
      std::array<int, OutN> out;
      auto ret = std::adjacent_difference(
          policy, Iter1(input.data()), Iter1(input.data() + input.size()), Iter2(out.data()), std::plus{});
      assert(base(ret) == out.data() + out.size());
      assert(out == plus_expected);
    }
  }

  template <class Policy>
  void operator()(Policy&& policy) {
    // simple test
    test<4>(policy, {1, 2, 3, 4}, {3, 5, 7}, {1, 1, 1});
    // empty range
    test<0, 0>(policy, {}, {}, {});
    // single element range
    test<1>(policy, {1}, {}, {});
    // two element range
    test<2>(policy, {10, 5}, {15}, {-5});

    // Large inputs with generated data
    for (auto e : {100, 322, 497, 2048}) {
      std::vector<int> input(e);
      std::iota(input.begin(), input.end(), 0);
      std::vector<int> expected(e - 1);
      auto binop = [](int lhs, int rhs) { return lhs + rhs * 3; };
      std::adjacent_difference(input.begin(), input.end(), expected.begin(), binop);
      std::vector<int> output(e - 1);
      std::adjacent_difference(input.begin(), input.end(), output.begin(), binop);
      assert(output == expected);
    }

    { // ensure that all values are used exactly once
      std::array input = {0, 1, 2, 3, 4, 5, 6, 7};
      std::array<bool, input.size() - 1> called{};
      std::array<int, input.size() - 1> output;
      std::adjacent_difference(input.data(), input.data() + input.size(), output.data(), [&](int lhs, int rhs) {
        assert(!called[rhs]);
        called[rhs] = true;
        return rhs - lhs;
      });
      assert(std::all_of(called.begin(), called.end(), [](bool b) { return b; }));
    }
  }
};

int main(int, char**) {
  types::for_each(types::forward_iterator_list<int*>{}, types::apply_type_identity{[](auto v1) {
                    using Iter1 = typename decltype(v1)::type;
                    types::for_each(
                        types::forward_iterator_list<int*>{},
                        TestIteratorWithPolicies<types::partial_instantiation<Test, Iter1>::template apply>{});
                  }});

  return 0;
}
