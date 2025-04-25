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

// template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2>
//   ForwardIterator2
//     rotate_copy(ExecutionPolicy&& exec,
//                 ForwardIterator1 first, ForwardIterator1 middle, ForwardIterator1 last,
//                 ForwardIterator2 result);

#include <algorithm>
#include <array>
#include <cassert>
#include <numeric>
#include <vector>

#include "test_macros.h"
#include "test_execution_policies.h"
#include "test_iterators.h"

template <class Iter>
struct Test {
  template <class Policy>
  void operator()(Policy&& policy) {
    { // simple test
      int in[] = {1, 2, 3, 4};
      int out[std::size(in)];

      decltype(auto) ret = std::rotate_copy(policy, Iter(in), Iter(in + 2), Iter(in + 4), Iter(out));
      static_assert(std::is_same_v<decltype(ret), Iter>);
      assert(base(ret) == out + 4);

      int expected[] = {3, 4, 1, 2};
      assert(std::equal(out, out + 4, expected));
    }
    { // rotating an empty range works
      std::array<int, 0> in  = {};
      std::array<int, 0> out = {};

      decltype(auto) ret =
          std::rotate_copy(policy, Iter(in.data()), Iter(in.data()), Iter(in.data()), Iter(out.data()));
      static_assert(std::is_same_v<decltype(ret), Iter>);
      assert(base(ret) == out.data());
    }
    { // rotating an single-element range works
      int in[] = {1};
      int out[std::size(in)];

      decltype(auto) ret = std::rotate_copy(policy, Iter(in), Iter(in), Iter(in + 1), Iter(out));
      static_assert(std::is_same_v<decltype(ret), Iter>);
      assert(base(ret) == out + 1);

      int expected[] = {1};
      assert(std::equal(out, out + 1, expected));
    }
    { // rotating a two-element range works
      int in[] = {1, 2};
      int out[std::size(in)];

      decltype(auto) ret = std::rotate_copy(policy, Iter(in), Iter(in + 1), Iter(in + 2), Iter(out));
      static_assert(std::is_same_v<decltype(ret), Iter>);
      assert(base(ret) == out + 2);

      int expected[] = {2, 1};
      assert(std::equal(out, out + 2, expected));
    }
    { // rotating a large range works
      std::vector<int> data(100);
      std::iota(data.begin(), data.end(), 0);
      for (int i = 0; i != 100; ++i) { // check all permutations
        auto copy = data;
        std::vector<int> out(100);
        std::rotate_copy(Iter(data.data()), Iter(data.data() + i), Iter(data.data() + data.size()), Iter(out.data()));
        assert(out[0] == i);
        assert(std::adjacent_find(out.begin(), out.end(), [](int lhs, int rhs) {
                 return lhs == 99 ? rhs != 0 : lhs != rhs - 1;
               }) == out.end());
        assert(copy == data);
      }
    }
  }
};

int main(int, char**) {
  types::for_each(types::forward_iterator_list<int*>{}, TestIteratorWithPolicies<Test>{});

  return 0;
}
