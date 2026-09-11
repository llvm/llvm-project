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
//           class BinaryOperation>
//   ForwardIterator2 adjacent_difference(ExecutionPolicy&& exec,
//                                        ForwardIterator1 first1,
//                                        ForwardIterator1 last1,
//                                        ForwardIterator2 first2,
//                                        BinaryOperation op);

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

EXECUTION_POLICY_SFINAE_TEST(adjacent_difference);

static_assert(sfinae_test_adjacent_difference<int, int*, int*, int*, int (*)(int, int)>);
static_assert(!sfinae_test_adjacent_difference<std::execution::parallel_policy, int*, int*, int*, int (*)(int, int)>);

// Types X and Y are provided to test adjacent_difference() against custom types.
// X is a source type that supports subtraction.
// Y is a destination type that can be assigned from X and compared for equality.

class X {
  int i_;
  X& operator=(const X&);

public:
  explicit X(int i) : i_(i) {}
  X(const X& x) : i_(x.i_) {}
  X& operator=(X&& x) {
    i_   = x.i_;
    x.i_ = -1;
    return *this;
  }
  friend class Y;
  friend struct Op;
};

class Y {
  int i_;
  Y& operator=(const Y&);

public:
  explicit Y(int i) : i_(i) {}
  Y(const Y& y) : i_(y.i_) {}
  void operator=(const X& x) { i_ = x.i_; }
  bool operator==(const Y& y) const { return i_ == y.i_; }
};

// ensures that the predicate is not equivalent to std::minus
struct Op {
  int operator()(int l, int r) const { return l * 2 - r; }
  X operator()(const X& l, const X& r) const { return X(l.i_ * 2 - r.i_); }
};

template <class Iter1, class Iter2>
struct Test {
  template <class ExecutionPolicy>
  void operator()(ExecutionPolicy&& policy) {
    {
      int ia[] = {10};
      int ir[] = {42};
      int ib[] = {42};
      Iter2 r =
          std::adjacent_difference(policy, Iter1(std::begin(ia)), Iter1(std::begin(ia)), Iter2(std::begin(ib)), Op{});
      assert(r == Iter2(std::begin(ib)));
      for (size_t i = 0; i < std::size(ia); ++i)
        assert(ib[i] == ir[i]);
    }
    {
      int ia[] = {10};
      int ir[] = {10};
      int ib[] = {42};
      Iter2 r =
          std::adjacent_difference(policy, Iter1(std::begin(ia)), Iter1(std::end(ia)), Iter2(std::begin(ib)), Op{});
      assert(r == Iter2(std::end(ib)));
      for (size_t i = 0; i < std::size(ia); ++i)
        assert(ib[i] == ir[i]);
    }
    {
      int ia[]              = {10, 30};
      int ir[]              = {10, 50};
      int ib[std::size(ia)] = {0};
      Iter2 r =
          std::adjacent_difference(policy, Iter1(std::begin(ia)), Iter1(std::end(ia)), Iter2(std::begin(ib)), Op{});
      assert(r == Iter2(std::end(ib)));
      for (size_t i = 0; i < std::size(ia); ++i)
        assert(ib[i] == ir[i]);
    }
    {
      int ia[]              = {15, 10, 6, 3, 1};
      int ir[]              = {15, 5, 2, 0, -1};
      int ib[std::size(ia)] = {0};
      Iter2 r =
          std::adjacent_difference(policy, Iter1(std::begin(ia)), Iter1(std::end(ia)), Iter2(std::begin(ib)), Op{});
      assert(r == Iter2(std::end(ib)));
      for (size_t i = 0; i < std::size(ia); ++i)
        assert(ib[i] == ir[i]);
    }
    {
      int ia[1073];
      int ib[1073];
      std::iota(std::begin(ia), std::end(ia), 1);
      runway_sample(std::size(ia), [&](size_t i) {
        Iter2 r = std::adjacent_difference(
            policy, Iter1(std::begin(ia)), Iter1(std::begin(ia) + i), Iter2(std::begin(ib)), Op{});
        assert(r == Iter2(std::begin(ib) + i));
        for (size_t j = 0; j < i; ++j)
          assert(ib[j] == (int)(j == 0 ? 1 : j + 2));
      });
    }
  }
};

template <class Iter1, class Iter2>
struct TestCustomTypes {
  template <class ExecutionPolicy>
  void operator()(ExecutionPolicy&& policy) {
    {
      X ia[3] = {X(1), X(5), X(8)};
      Y ir[3] = {Y(1), Y(9), Y(11)};
      Y ib[3] = {Y(0), Y(0), Y(0)};
      Iter2 r =
          std::adjacent_difference(policy, Iter1(std::begin(ia)), Iter1(std::end(ia)), Iter2(std::begin(ib)), Op{});
      assert(r == Iter2(std::end(ib)));
      for (size_t i = 0; i < std::size(ia); ++i)
        assert(ib[i] == ir[i]);
    }
  }
};

int main(int, char**) {
  types::for_each(
      types::concatenate_t<types::forward_iterator_list<int*>, types::forward_iterator_list<const int*>>{},
      types::apply_type_identity{[](auto v) {
        using Iter = typename decltype(v)::type;
        types::for_each(types::forward_iterator_list<int*>{},
                        TestIteratorWithPolicies<types::partial_instantiation<Test, Iter>::template apply>{});
      }});
  types::for_each(
      types::concatenate_t<types::forward_iterator_list<X*>, types::forward_iterator_list<const X*>>{},
      types::apply_type_identity{[](auto v) {
        using Iter = typename decltype(v)::type;
        types::for_each(
            types::forward_iterator_list<Y*>{},
            TestIteratorWithPolicies<types::partial_instantiation<TestCustomTypes, Iter>::template apply>{});
      }});
  return 0;
}
