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
//           class ForwardIterator2>
//   pair<ForwardIterator1, ForwardIterator2> mismatch(ExecutionPolicy&& exec,
//                                                     ForwardIterator1 first1,
//                                                     ForwardIterator1 last1,
//                                                     ForwardIterator2 first2);
//
// template <class ExecutionPolicy,
//           class ForwardIterator1,
//           class ForwardIterator2>
//   pair<ForwardIterator1, ForwardIterator2> mismatch(ExecutionPolicy&& exec,
//                                                     ForwardIterator1 first1,
//                                                     ForwardIterator1 last1,
//                                                     ForwardIterator2 first2,
//                                                     ForwardIterator2 last2);

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
#include "runway_sample.h"

EXECUTION_POLICY_SFINAE_TEST(mismatch);

static_assert(sfinae_test_mismatch<int, int*, int*, int*>);
static_assert(!sfinae_test_mismatch<std::execution::parallel_policy, int*, int*, int*>);
static_assert(sfinae_test_mismatch<int, int*, int*, int*, int*>);
static_assert(!sfinae_test_mismatch<std::execution::parallel_policy, int*, int*, int*, int*>);

// The types X and Y are provided to test that the mismatch algorithm can be used with heterogeneous custom types.

struct X {
  X() = delete;
  X(int i) : i_(i) {}
  X(const X&) = delete;
  int value() const { return i_; }

private:
  int i_;
};

struct Y {
  Y() = delete;
  Y(int i) : i_(i) {}
  Y(const Y&) = delete;
  int value() const { return i_; }

private:
  int i_;
};

bool operator==(const X& lhs, const Y& rhs) { return lhs.value() == rhs.value(); }

template <class Iter1, class Iter2>
struct Test {
  template <class ExecutionPolicy>
  void operator()(ExecutionPolicy&& policy) {
    {
      // check the return types
      int lhs[1]       = {0};
      int rhs[1]       = {0};
      auto res_3legged = std::mismatch(policy, Iter1(std::begin(lhs)), Iter1(std::begin(lhs)), Iter2(std::begin(rhs)));
      auto res_4legged = std::mismatch(
          policy, Iter1(std::begin(lhs)), Iter1(std::begin(lhs)), Iter2(std::begin(rhs)), Iter2(std::begin(rhs)));
      static_assert(std::is_same_v<decltype(res_3legged), std::pair<Iter1, Iter2>>);
      static_assert(std::is_same_v<decltype(res_4legged), std::pair<Iter1, Iter2>>);
    }
    {
      // empty ranges
      int lhs[1] = {0};
      int rhs[1] = {0};
      assert(std::mismatch(policy, Iter1(std::begin(lhs)), Iter1(std::begin(lhs)), Iter2(std::begin(rhs))) ==
             std::make_pair(Iter1(std::begin(lhs)), Iter2(std::begin(rhs))));
      assert(
          std::mismatch(
              policy, Iter1(std::begin(lhs)), Iter1(std::begin(lhs)), Iter2(std::begin(rhs)), Iter2(std::begin(rhs))) ==
          std::make_pair(Iter1(std::begin(lhs)), Iter2(std::begin(rhs))));
    }
    {
      // single element only
      int lhs[1] = {0};
      int rhs[1] = {0};
      assert(std::mismatch(policy, Iter1(std::begin(lhs)), Iter1(std::end(lhs)), Iter2(std::begin(rhs))) ==
             std::make_pair(Iter1(std::end(lhs)), Iter2(std::end(rhs))));
      assert(std::mismatch(
                 policy, Iter1(std::begin(lhs)), Iter1(std::end(lhs)), Iter2(std::begin(rhs)), Iter2(std::end(rhs))) ==
             std::make_pair(Iter1(std::end(lhs)), Iter2(std::end(rhs))));
    }
    { // same range without mismatch
      int lhs[8] = {0, 1, 2, 3, 0, 1, 2, 3};
      int rhs[8] = {0, 1, 2, 3, 0, 1, 2, 3};
      assert(std::mismatch(policy, Iter1(std::begin(lhs)), Iter1(std::end(lhs)), Iter2(std::begin(rhs))) ==
             std::make_pair(Iter1(std::end(lhs)), Iter2(std::end(rhs))));
      assert(std::mismatch(
                 policy, Iter1(std::begin(lhs)), Iter1(std::end(lhs)), Iter2(std::begin(rhs)), Iter2(std::end(rhs))) ==
             std::make_pair(Iter1(std::end(lhs)), Iter2(std::end(rhs))));
    }
    { // same range with mismatch
      int lhs[8] = {0, 1, 2, 2, 0, 1, 2, 3};
      int rhs[8] = {0, 1, 2, 3, 0, 1, 2, 3};
      assert(std::mismatch(policy, Iter1(std::begin(lhs)), Iter1(std::end(lhs)), Iter2(std::begin(rhs))) ==
             std::make_pair(Iter1(std::begin(lhs) + 3), Iter2(std::begin(rhs) + 3)));
      assert(std::mismatch(
                 policy, Iter1(std::begin(lhs)), Iter1(std::end(lhs)), Iter2(std::begin(rhs)), Iter2(std::end(rhs))) ==
             std::make_pair(Iter1(std::begin(lhs) + 3), Iter2(std::begin(rhs) + 3)));
    }
    { // second range is smaller
      int lhs[8] = {0, 1, 2, 2, 0, 1, 2, 3};
      int rhs[2] = {0, 1};
      assert(std::mismatch(
                 policy, Iter1(std::begin(lhs)), Iter1(std::end(lhs)), Iter2(std::begin(rhs)), Iter2(std::end(rhs))) ==
             std::make_pair(Iter1(std::begin(lhs) + 2), Iter2(std::begin(rhs) + 2)));
    }
    { // first range is smaller
      int lhs[2] = {0, 1};
      int rhs[8] = {0, 1, 2, 2, 0, 1, 2, 3};
      assert(std::mismatch(
                 policy, Iter1(std::begin(lhs)), Iter1(std::end(lhs)), Iter2(std::begin(rhs)), Iter2(std::end(rhs))) ==
             std::make_pair(Iter1(std::begin(lhs) + 2), Iter2(std::begin(rhs) + 2)));
    }
    { // same size, mismatching at various positions
      int lhs[1073];
      int rhs[1073];
      std::iota(std::begin(lhs), std::end(lhs), 0);
      std::copy(std::begin(lhs), std::end(lhs), std::begin(rhs));
      runway_sample(std::size(lhs), [&](size_t i) {
        lhs[i] = -1;
        assert(std::mismatch(policy, Iter1(std::begin(lhs)), Iter1(std::end(lhs)), Iter2(std::begin(rhs))) ==
               std::make_pair(Iter1(std::begin(lhs) + i), Iter2(std::begin(rhs) + i)));
        assert(
            std::mismatch(
                policy, Iter1(std::begin(lhs)), Iter1(std::end(lhs)), Iter2(std::begin(rhs)), Iter2(std::end(rhs))) ==
            std::make_pair(Iter1(std::begin(lhs) + i), Iter2(std::begin(rhs) + i)));
        lhs[i] = static_cast<int>(i);
      });
      assert(std::mismatch(policy, Iter1(std::begin(lhs)), Iter1(std::end(lhs)), Iter2(std::begin(rhs))) ==
             std::make_pair(Iter1(std::end(lhs)), Iter2(std::end(rhs))));
      assert(std::mismatch(
                 policy, Iter1(std::begin(lhs)), Iter1(std::end(lhs)), Iter2(std::begin(rhs)), Iter2(std::end(rhs))) ==
             std::make_pair(Iter1(std::end(lhs)), Iter2(std::end(rhs))));
    }
    { // same values, different lengths
      int lhs[739];
      int rhs[739];
      std::fill(std::begin(lhs), std::end(lhs), 42);
      std::fill(std::begin(rhs), std::end(rhs), 42);
      runway_sample(std::size(lhs), [&](size_t i) {
        // lhs is shorter
        assert(std::mismatch(
                   policy,
                   Iter1(std::begin(lhs)),
                   Iter1(std::begin(lhs) + i),
                   Iter2(std::begin(rhs)),
                   Iter2(std::end(rhs))) == std::make_pair(Iter1(std::begin(lhs) + i), Iter2(std::begin(rhs) + i)));
        // rhs is shorter
        assert(std::mismatch(policy,
                             Iter1(std::begin(lhs)),
                             Iter1(std::end(lhs)),
                             Iter2(std::begin(rhs)),
                             Iter2(std::begin(rhs) + i)) ==
               std::make_pair(Iter1(std::begin(lhs) + i), Iter2(std::begin(rhs) + i)));
      });
    }
  }
};

template <class Iter1, class Iter2>
struct TestXY {
  template <class ExecutionPolicy>
  void operator()(ExecutionPolicy&& policy) {
    { // same ranges
      X lhs[3] = {X(1), X(5), X(7)};
      Y rhs[3] = {Y(1), Y(5), Y(7)};
      assert(std::mismatch(policy, Iter1(std::begin(lhs)), Iter1(std::end(lhs)), Iter2(std::begin(rhs))) ==
             std::make_pair(Iter1(std::end(lhs)), Iter2(std::end(rhs))));
      assert(std::mismatch(
                 policy, Iter1(std::begin(lhs)), Iter1(std::end(lhs)), Iter2(std::begin(rhs)), Iter2(std::end(rhs))) ==
             std::make_pair(Iter1(std::end(lhs)), Iter2(std::end(rhs))));
    }
    { // one element mismatch
      X lhs[3] = {X(1), X(5), X(7)};
      Y rhs[3] = {Y(1), Y(5), Y(8)};
      assert(std::mismatch(policy, Iter1(std::begin(lhs)), Iter1(std::end(lhs)), Iter2(std::begin(rhs))) ==
             std::make_pair(Iter1(std::begin(lhs) + 2), Iter2(std::begin(rhs) + 2)));
      assert(std::mismatch(
                 policy, Iter1(std::begin(lhs)), Iter1(std::end(lhs)), Iter2(std::begin(rhs)), Iter2(std::end(rhs))) ==
             std::make_pair(Iter1(std::begin(lhs) + 2), Iter2(std::begin(rhs) + 2)));
    }
  }
};

int main(int, char**) {
  types::for_each(types::forward_iterator_list<const int*>{}, types::apply_type_identity{[](auto v) {
                    using Iter = typename decltype(v)::type;
                    types::for_each(
                        types::forward_iterator_list<const int*>{},
                        TestIteratorWithPolicies<types::partial_instantiation<Test, Iter>::template apply>{});
                  }});
  types::for_each(types::forward_iterator_list<const X*>{}, types::apply_type_identity{[](auto v) {
                    using Iter = typename decltype(v)::type;
                    types::for_each(
                        types::forward_iterator_list<const Y*>{},
                        TestIteratorWithPolicies<types::partial_instantiation<TestXY, Iter>::template apply>{});
                  }});
  return 0;
}
