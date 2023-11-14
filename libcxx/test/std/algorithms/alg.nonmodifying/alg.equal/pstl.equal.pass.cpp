//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// UNSUPPORETD: libcpp-has-no-incomplete-pstl

// <algorithm>

// template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2>
//   bool equal(ExecutionPolicy&& exec,
//              ForwardIterator1 first1, ForwardIterator1 last1,
//              ForwardIterator2 first2);
//
// template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2,
//          class BinaryPredicate>
//   bool equal(ExecutionPolicy&& exec,
//              ForwardIterator1 first1, ForwardIterator1 last1,
//              ForwardIterator2 first2, BinaryPredicate pred);
//
// template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2>
//   bool equal(ExecutionPolicy&& exec,
//              ForwardIterator1 first1, ForwardIterator1 last1,
//              ForwardIterator2 first2, ForwardIterator2 last2);
//
// template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2,
//          class BinaryPredicate>
//   bool equal(ExecutionPolicy&& exec,
//              ForwardIterator1 first1, ForwardIterator1 last1,
//              ForwardIterator2 first2, ForwardIterator2 last2,
//              BinaryPredicate pred);

#include <algorithm>
#include <cassert>

#include "test_execution_policies.h"
#include "test_iterators.h"
#include "test_macros.h"

template <class Iter1, class Iter2>
struct Test {
  template <class Policy>
  void operator()(Policy&& policy) {
    { // 3 iter overloads
      int a[] = {1, 2, 3, 4, 5, 6, 7, 8};
      int b[] = {1, 2, 3, 5, 6, 7, 8, 9};

      // simple test
      assert(std::equal(policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(a))));

      // check that false is returned on different ranges
      assert(!std::equal(policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(b))));

      // check that the predicate is used
      assert(std::equal(policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(b)), [](int lhs, int rhs) {
        return lhs == rhs || lhs + 1 == rhs || rhs + 1 == lhs;
      }));
    }

    { // 4 iter overloads
      int a[] = {1, 2, 3, 4, 5, 6, 7, 8};
      int b[] = {1, 2, 3, 5, 6, 7, 8, 9};

      // simple test
      assert(std::equal(policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(a)), Iter2(std::end(a))));

      // check that false is returned on different ranges
      assert(!std::equal(policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(b)), Iter2(std::end(b))));

      // check that false is returned on different sized ranges
      assert(
          !std::equal(policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(a)), Iter2(std::end(a) - 1)));

      // check that the predicate is used
      assert(std::equal(
          policy,
          Iter1(std::begin(a)),
          Iter1(std::end(a)),
          Iter2(std::begin(b)),
          Iter2(std::end(b)),
          [](int lhs, int rhs) { return lhs == rhs || lhs + 1 == rhs || rhs + 1 == lhs; }));
    }
  }
};

int main(int, char**) {
  types::for_each(types::forward_iterator_list<int*>{}, types::apply_type_identity{[](auto v) {
                    using Iter1 = typename decltype(v)::type;
                    types::for_each(
                        types::forward_iterator_list<int*>{},
                        TestIteratorWithPolicies<types::partial_instantiation<Test, Iter1>::template apply>{});
                  }});
}
