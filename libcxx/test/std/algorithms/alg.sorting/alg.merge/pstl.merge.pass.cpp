//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: libcpp-has-no-incomplete-pstl

// template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2,
//          class ForwardIterator>
//   ForwardIterator
//     merge(ExecutionPolicy&& exec,
//           ForwardIterator1 first1, ForwardIterator1 last1,
//           ForwardIterator2 first2, ForwardIterator2 last2,
//           ForwardIterator result);
//
// template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2,
//          class ForwardIterator, class Compare>
//   ForwardIterator
//     merge(ExecutionPolicy&& exec,
//           ForwardIterator1 first1, ForwardIterator1 last1,
//           ForwardIterator2 first2, ForwardIterator2 last2,
//           ForwardIterator result, Compare comp);

#include <algorithm>
#include <array>
#include <cassert>
#include <iterator>
#include <numeric>
#include <vector>

#include "type_algorithms.h"
#include "test_execution_policies.h"
#include "test_iterators.h"

template <class Iter1, class Iter2>
struct Test {
  template <class Policy>
  void operator()(Policy&& policy) {
    { // simple test
      int a[] = {1, 3, 5, 7, 9};
      int b[] = {2, 4, 6, 8, 10};
      std::array<int, std::size(a) + std::size(b)> out;
      std::merge(
          policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(b)), Iter2(std::end(b)), std::begin(out));
      assert((out == std::array{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}));
    }

    { // check that it works with both ranges being empty
      std::array<int, 0> a;
      std::array<int, 0> b;
      std::array<int, std::size(a) + std::size(b)> out;
      std::merge(policy,
                 Iter1(std::data(a)),
                 Iter1(std::data(a) + std::size(a)),
                 Iter2(std::data(b)),
                 Iter2(std::data(b) + std::size(b)),
                 std::begin(out));
    }
    { // check that it works with the first range being empty
      std::array<int, 0> a;
      int b[] = {2, 4, 6, 8, 10};
      std::array<int, std::size(a) + std::size(b)> out;
      std::merge(policy,
                 Iter1(std::data(a)),
                 Iter1(std::data(a) + std::size(a)),
                 Iter2(std::begin(b)),
                 Iter2(std::end(b)),
                 std::begin(out));
      assert((out == std::array{2, 4, 6, 8, 10}));
    }

    { // check that it works with the second range being empty
      int a[] = {2, 4, 6, 8, 10};
      std::array<int, 0> b;
      std::array<int, std::size(a) + std::size(b)> out;
      std::merge(policy,
                 Iter1(std::begin(a)),
                 Iter1(std::end(a)),
                 Iter2(std::data(b)),
                 Iter2(std::data(b) + std::size(b)),
                 std::begin(out));
      assert((out == std::array{2, 4, 6, 8, 10}));
    }

    { // check that it works when the ranges don't have the same length
      int a[] = {2, 4, 6, 8, 10};
      int b[] = {3, 4};
      std::array<int, std::size(a) + std::size(b)> out;
      std::merge(
          policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(b)), Iter2(std::end(b)), std::begin(out));
      assert((out == std::array{2, 3, 4, 4, 6, 8, 10}));
    }

    { // check that large ranges work
      std::vector<int> a(100);
      std::vector<int> b(100);
      {
        int i = 0;
        for (auto& e : a) {
          e = i;
          i += 2;
        }
      }

      {
        int i = 1;
        for (auto& e : b) {
          e = i;
          i += 2;
        }
      }

      std::vector<int> out(std::size(a) + std::size(b));
      std::merge(policy,
                 Iter1(a.data()),
                 Iter1(a.data() + a.size()),
                 Iter2(b.data()),
                 Iter2(b.data() + b.size()),
                 std::begin(out));
      std::vector<int> expected(200);
      std::iota(expected.begin(), expected.end(), 0);
      assert(std::equal(out.begin(), out.end(), expected.begin()));
    }

    { // check that the predicate is used
      int a[] = {10, 9, 8, 7};
      int b[] = {8, 4, 3};
      std::array<int, std::size(a) + std::size(b)> out;
      std::merge(
          policy,
          Iter1(std::begin(a)),
          Iter1(std::end(a)),
          Iter2(std::begin(b)),
          Iter2(std::end(b)),
          std::begin(out),
          std::greater{});
      assert((out == std::array{10, 9, 8, 8, 7, 4, 3}));
    }
  }
};

int main(int, char**) {
  types::for_each(types::forward_iterator_list<int*>{}, types::apply_type_identity{[](auto v) {
                    using Iter = typename decltype(v)::type;
                    types::for_each(
                        types::forward_iterator_list<int*>{},
                        TestIteratorWithPolicies<types::partial_instantiation<Test, Iter>::template apply>{});
                  }});

  return 0;
}
