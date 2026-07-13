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
//           class ForwardIterator2,
//           class BinaryPredicate>
//   pair<ForwardIterator1, ForwardIterator2> mismatch(ExecutionPolicy&& exec,
//                                                     ForwardIterator1 first1,
//                                                     ForwardIterator1 last1,
//                                                     ForwardIterator2 first2,
//                                                     BinaryPredicate pred);
//
// template <class ExecutionPolicy,
//           class ForwardIterator1,
//           class ForwardIterator2,
//           class BinaryPredicate>
//   pair<ForwardIterator1, ForwardIterator2> mismatch(ExecutionPolicy&& exec,
//                                                     ForwardIterator1 first1,
//                                                     ForwardIterator1 last1,
//                                                     ForwardIterator2 first2,
//                                                     ForwardIterator2 last2,
//                                                     BinaryPredicate pred);

#include <algorithm>
#include <array>
#include <cassert>
#include <functional>
#include <iterator>
#include <limits>
#include <numeric>

#include "test_execution_policies.h"
#include "test_iterators.h"
#include "test_macros.h"
#include "type_algorithms.h"

EXECUTION_POLICY_SFINAE_TEST(mismatch);

static_assert(sfinae_test_mismatch<int, int*, int*, int*, bool (*)(int, int)>);
static_assert(!sfinae_test_mismatch<std::execution::parallel_policy, int*, int*, int*, bool (*)(int, int)>);
static_assert(sfinae_test_mismatch<int, int*, int*, int*, int*, bool (*)(int, int)>);
static_assert(!sfinae_test_mismatch<std::execution::parallel_policy, int*, int*, int*, int*, bool (*)(int, int)>);

// TODO: switch with a shared implemented once it's merged into main
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

// The types X and Y are provided to test that the mismatch algorithm can be used with heterogeneous custom types.

struct X {
  X() = delete;
  X(int i) : i(i) {}
  X(const X&) = delete;
  int value() const { return i; }

private:
  int i;
};

struct Y {
  Y() = delete;
  Y(int i) : i(i) {}
  Y(const Y&) = delete;
  int value() const { return i; }

private:
  int i;
};

struct Pred {
  bool operator()(int lhs, int rhs) const { return lhs * 2 == rhs; }
  bool operator()(const X& lhs, const Y& rhs) const { return lhs.value() * 2 == rhs.value(); }
};

template <class Iter1, class Iter2>
struct Test {
  template <class ExecutionPolicy>
  void operator()(ExecutionPolicy&& policy) {
    {
      // empty ranges
      std::array<int, 1> lhs = {0};
      std::array<int, 1> rhs = {0};
      assert(std::mismatch(policy, Iter1(lhs.begin()), Iter1(lhs.begin()), Iter2(rhs.begin()), Pred{}) ==
             std::make_pair(Iter1(lhs.begin()), Iter2(rhs.begin())));
      assert(std::mismatch(
                 policy, Iter1(lhs.begin()), Iter1(lhs.begin()), Iter2(rhs.begin()), Iter2(rhs.begin()), Pred{}) ==
             std::make_pair(Iter1(lhs.begin()), Iter2(rhs.begin())));
    }
    {
      // single element only
      std::array<int, 1> lhs = {1};
      std::array<int, 1> rhs = {2};
      assert(std::mismatch(policy, Iter1(lhs.begin()), Iter1(lhs.end()), Iter2(rhs.begin()), Pred{}) ==
             std::make_pair(Iter1(lhs.end()), Iter2(rhs.end())));
      assert(
          std::mismatch(policy, Iter1(lhs.begin()), Iter1(lhs.end()), Iter2(rhs.begin()), Iter2(rhs.end()), Pred{}) ==
          std::make_pair(Iter1(lhs.end()), Iter2(rhs.end())));
    }
    { // same range without mismatch
      std::array<int, 8> lhs = {0, 1, 2, 3, 0, 1, 2, 3};
      std::array<int, 8> rhs = {0, 2, 4, 6, 0, 2, 4, 6};
      assert(std::mismatch(policy, Iter1(lhs.begin()), Iter1(lhs.end()), Iter2(rhs.begin()), Pred{}) ==
             std::make_pair(Iter1(lhs.end()), Iter2(rhs.end())));
      assert(
          std::mismatch(policy, Iter1(lhs.begin()), Iter1(lhs.end()), Iter2(rhs.begin()), Iter2(rhs.end()), Pred{}) ==
          std::make_pair(Iter1(lhs.end()), Iter2(rhs.end())));
    }
    { // same range with mismatch
      std::array<int, 8> lhs = {0, 1, 2, 3, 0, 1, 2, 3};
      std::array<int, 8> rhs = {0, 2, 4, 7, 0, 2, 4, 6};
      assert(std::mismatch(policy, Iter1(lhs.begin()), Iter1(lhs.end()), Iter2(rhs.begin()), Pred{}) ==
             std::make_pair(Iter1(lhs.begin() + 3), Iter2(rhs.begin() + 3)));
      assert(
          std::mismatch(policy, Iter1(lhs.begin()), Iter1(lhs.end()), Iter2(rhs.begin()), Iter2(rhs.end()), Pred{}) ==
          std::make_pair(Iter1(lhs.begin() + 3), Iter2(rhs.begin() + 3)));
    }
    { // second range is smaller
      std::array<int, 8> lhs = {0, 1, 2, 2, 0, 1, 2, 3};
      std::array<int, 2> rhs = {0, 2};
      assert(
          std::mismatch(policy, Iter1(lhs.begin()), Iter1(lhs.end()), Iter2(rhs.begin()), Iter2(rhs.end()), Pred{}) ==
          std::make_pair(Iter1(lhs.begin() + 2), Iter2(rhs.begin() + 2)));
    }
    { // first range is smaller
      std::array<int, 2> lhs = {0, 1};
      std::array<int, 8> rhs = {0, 2, 2, 2, 0, 1, 2, 3};
      assert(
          std::mismatch(policy, Iter1(lhs.begin()), Iter1(lhs.end()), Iter2(rhs.begin()), Iter2(rhs.end()), Pred{}) ==
          std::make_pair(Iter1(lhs.begin() + 2), Iter2(rhs.begin() + 2)));
    }
    { // same size, mismatching at various positions
      std::array<int, 1073> lhs;
      std::array<int, 1073> rhs;
      std::iota(std::begin(lhs), std::end(lhs), 0);
      rhs = lhs;
      std::for_each(std::begin(rhs), std::end(rhs), [](int& x) { x *= 2; });
      runway_sample(lhs.size(), [&](size_t i) {
        lhs[i] = -1;
        assert(std::mismatch(policy, Iter1(lhs.begin()), Iter1(lhs.end()), Iter2(rhs.begin()), Pred{}) ==
               std::make_pair(Iter1(lhs.begin() + i), Iter2(rhs.begin() + i)));
        assert(
            std::mismatch(policy, Iter1(lhs.begin()), Iter1(lhs.end()), Iter2(rhs.begin()), Iter2(rhs.end()), Pred{}) ==
            std::make_pair(Iter1(lhs.begin() + i), Iter2(rhs.begin() + i)));
        lhs[i] = i;
      });
      assert(std::mismatch(policy, Iter1(lhs.begin()), Iter1(lhs.end()), Iter2(rhs.begin()), Pred{}) ==
             std::make_pair(Iter1(lhs.end()), Iter2(rhs.end())));
      assert(
          std::mismatch(policy, Iter1(lhs.begin()), Iter1(lhs.end()), Iter2(rhs.begin()), Iter2(rhs.end()), Pred{}) ==
          std::make_pair(Iter1(lhs.end()), Iter2(rhs.end())));
    }
    { // same values, different lengths
      std::array<int, 739> lhs;
      std::array<int, 739> rhs;
      lhs.fill(42);
      rhs.fill(84);
      runway_sample(lhs.size(), [&](size_t i) {
        // lhs is shorter
        assert(std::mismatch(
                   policy, Iter1(lhs.begin()), Iter1(lhs.begin() + i), Iter2(rhs.begin()), Iter2(rhs.end()), Pred{}) ==
               std::make_pair(Iter1(lhs.begin() + i), Iter2(rhs.begin() + i)));
        // rhs is shorter
        assert(std::mismatch(
                   policy, Iter1(lhs.begin()), Iter1(lhs.end()), Iter2(rhs.begin()), Iter2(rhs.begin() + i), Pred{}) ==
               std::make_pair(Iter1(lhs.begin() + i), Iter2(rhs.begin() + i)));
      });
    }
  }
};

template <class Iter1, class Iter2>
struct TestXY {
  template <class ExecutionPolicy>
  void operator()(ExecutionPolicy&& policy) {
    { // same ranges
      std::array<X, 3> lhs = {X(1), X(5), X(7)};
      std::array<Y, 3> rhs = {Y(2), Y(10), Y(14)};
      assert(std::mismatch(policy, Iter1(lhs.begin()), Iter1(lhs.end()), Iter2(rhs.begin()), Pred{}) ==
             std::make_pair(Iter1(lhs.end()), Iter2(rhs.end())));
      assert(
          std::mismatch(policy, Iter1(lhs.begin()), Iter1(lhs.end()), Iter2(rhs.begin()), Iter2(rhs.end()), Pred{}) ==
          std::make_pair(Iter1(lhs.end()), Iter2(rhs.end())));
    }
    { // one element mismatch
      std::array<X, 3> lhs = {X(1), X(5), X(7)};
      std::array<Y, 3> rhs = {Y(2), Y(10), Y(13)};
      assert(std::mismatch(policy, Iter1(lhs.begin()), Iter1(lhs.end()), Iter2(rhs.begin()), Pred{}) ==
             std::make_pair(Iter1(lhs.begin() + 2), Iter2(rhs.begin() + 2)));
      assert(
          std::mismatch(policy, Iter1(lhs.begin()), Iter1(lhs.end()), Iter2(rhs.begin()), Iter2(rhs.end()), Pred{}) ==
          std::make_pair(Iter1(lhs.begin() + 2), Iter2(rhs.begin() + 2)));
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
