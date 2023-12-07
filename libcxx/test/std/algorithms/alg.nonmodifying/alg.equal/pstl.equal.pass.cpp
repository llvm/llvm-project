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
#include <iterator>

#include "test_execution_policies.h"
#include "test_iterators.h"
#include "test_macros.h"

template <class It1, class It2>
struct Test {
  template <class Policy>
  void operator()(Policy&& policy) {
    { // 3 iter overloads
      // check with equal ranges
      {
        int a[] = {1, 2, 3, 4};
        int b[] = {1, 2, 3, 4};
        assert(std::equal(policy, It1(std::begin(a)), It1(std::end(a)), It2(std::begin(b))));
      }

      // check with an empty range
      {
        int a[] = {999};
        int b[] = {1, 2, 3};
        assert(std::equal(policy, It1(std::begin(a)), It1(std::begin(a)), It2(std::begin(b))));
      }

      // check with different ranges
      {
        int a[] = {1, 2, 3};
        int b[] = {3, 2, 1};
        assert(!std::equal(policy, It1(std::begin(a)), It1(std::end(a)), It2(std::begin(b))));
      }

      // check that the predicate is used
      {
        int a[] = {2, 4, 6, 8, 10};
        int b[] = {12, 14, 16, 18, 20};
        assert(std::equal(policy, It1(std::begin(a)), It1(std::end(a)), It2(std::begin(b)), [](int lhs, int rhs) {
          return lhs % 2 == rhs % 2;
        }));
      }
    }

    { // 4 iter overloads
      // check with equal ranges of equal size
      {
        int a[] = {1, 2, 3, 4};
        int b[] = {1, 2, 3, 4};
        assert(std::equal(policy, It1(std::begin(a)), It1(std::end(a)), It2(std::begin(b)), It2(std::end(b))));
      }

      // check with unequal ranges of equal size
      {
        int a[] = {1, 2, 3, 4};
        int b[] = {4, 3, 2, 1};
        assert(!std::equal(policy, It1(std::begin(a)), It1(std::end(a)), It2(std::begin(b)), It2(std::end(b))));
      }

      // check with equal ranges of unequal size
      {
        {
          int a[] = {1, 2, 3, 4};
          int b[] = {1, 2, 3, 4, 5};
          assert(!std::equal(policy, It1(std::begin(a)), It1(std::end(a)), It2(std::begin(b)), It2(std::end(b))));
        }
        {
          int a[] = {1, 2, 3, 4, 5};
          int b[] = {1, 2, 3, 4};
          assert(!std::equal(policy, It1(std::begin(a)), It1(std::end(a)), It2(std::begin(b)), It2(std::end(b))));
        }
      }

      // check empty ranges
      {
        // empty/empty
        {
          int a[] = {888};
          int b[] = {999};
          assert(std::equal(policy, It1(std::begin(a)), It1(std::begin(a)), It2(std::begin(b)), It2(std::begin(b))));
        }
        // empty/non-empty
        {
          int a[] = {999};
          int b[] = {999};
          assert(!std::equal(policy, It1(std::begin(a)), It1(std::begin(a)), It2(std::begin(b)), It2(std::end(b))));
        }
        // non-empty/empty
        {
          int a[] = {999};
          int b[] = {999};
          assert(!std::equal(policy, It1(std::begin(a)), It1(std::end(a)), It2(std::begin(b)), It2(std::begin(b))));
        }
      }

      // check that the predicate is used
      {
        int a[] = {2, 4, 6, 8, 10};
        int b[] = {12, 14, 16, 18, 20};
        assert(std::equal(
            policy, It1(std::begin(a)), It1(std::end(a)), It2(std::begin(b)), It2(std::end(b)), [](int lhs, int rhs) {
              return lhs % 2 == rhs % 2;
            }));
      }
    }
  }
};

int main(int, char**) {
  types::for_each(types::forward_iterator_list<int*>{}, types::apply_type_identity{[](auto v) {
                    using It1 = typename decltype(v)::type;
                    types::for_each(
                        types::forward_iterator_list<int*>{},
                        TestIteratorWithPolicies<types::partial_instantiation<Test, It1>::template apply>{});
                  }});
  return 0;
}
