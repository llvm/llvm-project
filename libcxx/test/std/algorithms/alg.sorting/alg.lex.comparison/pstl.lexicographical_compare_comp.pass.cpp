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
//           class Compare>
//   bool lexicographical_compare(ExecutionPolicy&& exec,
//                                ForwardIterator1 first1,
//                                ForwardIterator1 last1,
//                                ForwardIterator2 first2,
//                                ForwardIterator2 last2,
//                                Compare comp);

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

EXECUTION_POLICY_SFINAE_TEST(lexicographical_compare);

static_assert(sfinae_test_lexicographical_compare<int, int*, int*, int*, int*, bool (*)(int, int)>);
static_assert(
    !sfinae_test_lexicographical_compare<std::execution::parallel_policy, int*, int*, int*, int*, bool (*)(int, int)>);

// The types X and Y are provided to test that lexicographical_compare can be used with heterogeneous custom types.

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

struct Comp {
  bool operator()(int lhs, int rhs) const { return lhs > rhs; }
  bool operator()(const X& lhs, const Y& rhs) const { return lhs.value() > rhs.value(); }
  bool operator()(const Y& lhs, const X& rhs) const { return lhs.value() > rhs.value(); }
};

template <class Iter1, class Iter2>
struct Test {
  template <class ExecutionPolicy>
  void operator()(ExecutionPolicy&& policy) {
    { // check the return types
      int lhs[] = {0};
      int rhs[] = {0};
      auto res  = std::lexicographical_compare(
          policy,
          Iter1(std::begin(lhs)),
          Iter1(std::begin(lhs)),
          Iter2(std::begin(rhs)),
          Iter2(std::begin(rhs)),
          Comp{});
      static_assert(std::is_same_v<decltype(res), bool>);
    }
    { // 2 empty ranges
      int lhs[] = {0};
      int rhs[] = {0};
      assert(!std::lexicographical_compare(
          policy,
          Iter1(std::begin(lhs)),
          Iter1(std::begin(lhs)),
          Iter2(std::begin(rhs)),
          Iter2(std::begin(rhs)),
          Comp{}));
    }
    { // left empty, right non-empty
      int lhs[] = {0};
      int rhs[] = {0};
      assert(std::lexicographical_compare(
          policy,
          Iter1(std::begin(lhs)),
          Iter1(std::begin(lhs)),
          Iter2(std::begin(rhs)),
          Iter2(std::end(rhs)),
          Comp{}));
    }
    { // left non-empty, right empty
      int lhs[] = {0};
      int rhs[] = {0};
      assert(!std::lexicographical_compare(
          policy,
          Iter1(std::begin(lhs)),
          Iter1(std::end(lhs)),
          Iter2(std::begin(rhs)),
          Iter2(std::begin(rhs)),
          Comp{}));
    }
    { // same size, same single element
      int lhs[] = {0};
      int rhs[] = {0};
      assert(!std::lexicographical_compare(
          policy, Iter1(std::begin(lhs)), Iter1(std::end(lhs)), Iter2(std::begin(rhs)), Iter2(std::end(rhs)), Comp{}));
    }
    { // left longer
      int lhs[] = {1, 2, 3, 4, 5};
      int rhs[] = {1, 2, 3};
      assert(!std::lexicographical_compare(
          policy, Iter1(std::begin(lhs)), Iter1(std::end(lhs)), Iter2(std::begin(rhs)), Iter2(std::end(rhs)), Comp{}));
    }
    { // right longer
      int lhs[] = {1, 2, 3};
      int rhs[] = {1, 2, 3, 4, 5};
      assert(std::lexicographical_compare(
          policy, Iter1(std::begin(lhs)), Iter1(std::end(lhs)), Iter2(std::begin(rhs)), Iter2(std::end(rhs)), Comp{}));
    }
    { // same size, left is lexicographically less than right
      int lhs[] = {1, 2, 4, 4, 5};
      int rhs[] = {1, 2, 3, 4, 5};
      assert(std::lexicographical_compare(
          policy, Iter1(std::begin(lhs)), Iter1(std::end(lhs)), Iter2(std::begin(rhs)), Iter2(std::end(rhs)), Comp{}));
    }
    { // same size, right is lexicographically less than left
      int lhs[] = {1, 2, 3, 4, 5};
      int rhs[] = {1, 2, 4, 4, 5};
      assert(!std::lexicographical_compare(
          policy, Iter1(std::begin(lhs)), Iter1(std::end(lhs)), Iter2(std::begin(rhs)), Iter2(std::end(rhs)), Comp{}));
    }
    { // same size, different at various positions
      int lhs[1073];
      int rhs[1073];
      std::iota(std::begin(lhs), std::end(lhs), 0);
      std::copy(std::begin(lhs), std::end(lhs), std::begin(rhs));
      runway_sample(std::size(lhs), [&](size_t i) {
        lhs[i] = 10'000;
        assert(std::lexicographical_compare(
            policy,
            Iter1(std::begin(lhs)),
            Iter1(std::end(lhs)),
            Iter2(std::begin(rhs)),
            Iter2(std::end(rhs)),
            Comp{}));
        lhs[i] = static_cast<int>(i);
        rhs[i] = 10'000;
        assert(!std::lexicographical_compare(
            policy,
            Iter1(std::begin(lhs)),
            Iter1(std::end(lhs)),
            Iter2(std::begin(rhs)),
            Iter2(std::end(rhs)),
            Comp{}));
        rhs[i] = static_cast<int>(i);
      });
      assert(!std::lexicographical_compare(
          policy, Iter1(std::begin(lhs)), Iter1(std::end(lhs)), Iter2(std::begin(rhs)), Iter2(std::end(rhs)), Comp{}));
    }
  }
};

template <class Iter1, class Iter2>
struct TestXY {
  template <class ExecutionPolicy>
  void operator()(ExecutionPolicy&& policy) {
    { // same ranges
      X lhs[] = {X(1), X(5), X(7)};
      Y rhs[] = {Y(1), Y(5), Y(7)};
      assert(!std::lexicographical_compare(
          policy, Iter1(std::begin(lhs)), Iter1(std::end(lhs)), Iter2(std::begin(rhs)), Iter2(std::end(rhs)), Comp{}));
    }
    { // one element difference
      X lhs[] = {X(1), X(5), X(8)};
      Y rhs[] = {Y(1), Y(5), Y(7)};
      assert(std::lexicographical_compare(
          policy, Iter1(std::begin(lhs)), Iter1(std::end(lhs)), Iter2(std::begin(rhs)), Iter2(std::end(rhs)), Comp{}));
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
