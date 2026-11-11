//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++17

// UNSUPPORTED: libcpp-has-no-incomplete-pstl

// <algorithm>

// template <class ExecutionPolicy,
//           class ForwardIterator1,
//           class ForwardIterator2>
//   ForwardIterator2 swap_ranges(ExecutionPolicy&& exec,
//                                ForwardIterator1 first1, ForwardIterator1 last1,
//                                ForwardIterator2 last2);

#include <array>
#include <algorithm>
#include <cassert>
#include <cstring>
#include <functional>
#include <iterator>
#include <memory>
#include <numeric>

#include "test_execution_policies.h"
#include "test_iterators.h"
#include "test_macros.h"
#include "type_algorithms.h"
#include "runway_sample.h"

EXECUTION_POLICY_SFINAE_TEST(swap_ranges);

static_assert(sfinae_test_swap_ranges<int, int*, int*, int*>);
static_assert(!sfinae_test_swap_ranges<std::execution::parallel_policy, int*, int*, int*>);

template <class Iter1, class Iter2>
struct TestInt {
  template <class ExecutionPolicy>
  void operator()(ExecutionPolicy&& policy) {
    { // Check the return type
      int a[]  = {0};
      int b[]  = {0};
      auto ret = std::swap_ranges(policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(b)));
      ASSERT_SAME_TYPE(decltype(ret), Iter2);
      assert(ret == Iter2(std::end(b)));
    }
    { // Size=1
      int a[] = {1};
      int b[] = {2};
      std::swap_ranges(policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(b)));
      assert(a[0] == 2);
      assert(b[0] == 1);
    }
    { // Size=2
      int a[] = {1, 2};
      int b[] = {4, 5};
      std::swap_ranges(policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(b)));
      assert(a[0] == 4 && a[1] == 5);
      assert(b[0] == 1 && b[1] == 2);
    }
    { // Size=3
      std::array<int, 3> a = {1, 2, 3}, a0 = a;
      std::array<int, 3> b = {4, 5, 6}, b0 = b;
      std::swap_ranges(policy, Iter1(a.data()), Iter1(a.data() + a.size()), Iter2(b.data()));
      assert(a == b0);
      assert(b == a0);
    }
    { // Size=4
      std::array<int, 4> a = {1, 2, 3, 4}, a0 = a;
      std::array<int, 4> b = {5, 6, 7, 8}, b0 = b;
      std::swap_ranges(policy, Iter1(a.data()), Iter1(a.data() + a.size()), Iter2(b.data()));
      assert(a == b0);
      assert(b == a0);
    }
    { // Size=5
      std::array<int, 5> a = {1, 2, 3, 4, 5}, a0 = a;
      std::array<int, 5> b = {6, 7, 8, 9, 10}, b0 = b;
      std::swap_ranges(policy, Iter1(a.data()), Iter1(a.data() + a.size()), Iter2(b.data()));
      assert(a == b0);
      assert(b == a0);
    }
    { // Different sampled sizes
      constexpr size_t n = 1073;
      std::array<int, n> a0;
      std::array<int, n> b0;
      std::iota(a0.begin(), a0.end(), 0);     // a0 is     0,     1,     2, ...
      std::iota(b0.begin(), b0.end(), 10000); // b0 is 10000, 10001, 10002, ...
      runway_sample(n + 1, [&](size_t size) {
        std::array<int, n> a;
        std::array<int, n> b;
        // Copy the first 'size' elements and fill the rest with sentinel values
        std::copy_n(a0.begin(), size, a.begin());
        std::fill(a.begin() + size, a.end(), -1);
        std::copy_n(b0.begin(), size, b.begin());
        std::fill(b.begin() + size, b.end(), -2);
        auto ret = std::swap_ranges(policy, Iter1(a.data()), Iter1(a.data() + size), Iter2(b.data()));
        assert(ret == Iter2(b.data() + size));
        // Check that the first 'size' elements have been swapped correctly, and the rest remain as sentinel values
        assert(std::equal(a.begin(), a.begin() + size, b0.begin()));
        assert(std::all_of(a.begin() + size, a.end(), [](int x) { return x == -1; }));
        assert(std::equal(b.begin(), b.begin() + size, a0.begin()));
        assert(std::all_of(b.begin() + size, b.end(), [](int x) { return x == -2; }));
      });
    }
  }
};

struct SwappableNotMovable {
  SwappableNotMovable(int value) : value_(value) {}
  SwappableNotMovable(const SwappableNotMovable&)            = delete;
  SwappableNotMovable& operator=(const SwappableNotMovable&) = delete;
  int value() const { return value_; }
  friend void swap(SwappableNotMovable& lhs, SwappableNotMovable& rhs) noexcept { std::swap(lhs.value_, rhs.value_); }

private:
  int value_;
};

template <class Iter1, class Iter2>
struct TestSwappableNotMovable {
  template <class ExecutionPolicy>
  void operator()(ExecutionPolicy&& policy) {
    SwappableNotMovable a[] = {1, 2, 3};
    SwappableNotMovable b[] = {4, 5, 6};
    std::swap_ranges(policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(b)));
    assert(a[0].value() == 4 && a[1].value() == 5 && a[2].value() == 6);
    assert(b[0].value() == 1 && b[1].value() == 2 && b[2].value() == 3);
  }
};

struct MovableNotCopiable {
  MovableNotCopiable(int value) : value_(value) {}
  MovableNotCopiable(const MovableNotCopiable&)            = delete;
  MovableNotCopiable& operator=(const MovableNotCopiable&) = delete;
  MovableNotCopiable(MovableNotCopiable&& other) : value_(other.value_) { other.value_ = -1; }
  MovableNotCopiable& operator=(MovableNotCopiable&& other) {
    value_       = other.value_;
    other.value_ = -1;
    return *this;
  }
  int value() const { return value_; }

private:
  int value_;
};

template <class Iter1, class Iter2>
struct TestMovableNotCopiable {
  template <class ExecutionPolicy>
  void operator()(ExecutionPolicy&& policy) {
    MovableNotCopiable a[] = {1, 2, 3};
    MovableNotCopiable b[] = {4, 5, 6};
    std::swap_ranges(policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(b)));
    assert(a[0].value() == 4 && a[1].value() == 5 && a[2].value() == 6);
    assert(b[0].value() == 1 && b[1].value() == 2 && b[2].value() == 3);
  }
};

int main(int, char**) {
  types::for_each(types::forward_iterator_list<int*>{}, types::apply_type_identity{[](auto v) {
                    using Iter = typename decltype(v)::type;
                    types::for_each(
                        types::forward_iterator_list<int*>{},
                        TestIteratorWithPolicies<types::partial_instantiation<TestInt, Iter>::template apply>{});
                  }});
  types::for_each(
      types::forward_iterator_list<SwappableNotMovable*>{}, types::apply_type_identity{[](auto v) {
        using Iter = typename decltype(v)::type;
        types::for_each(
            types::forward_iterator_list<SwappableNotMovable*>{},
            TestIteratorWithPolicies<types::partial_instantiation<TestSwappableNotMovable, Iter>::template apply>{});
      }});
  types::for_each(
      types::forward_iterator_list<MovableNotCopiable*>{}, types::apply_type_identity{[](auto v) {
        using Iter = typename decltype(v)::type;
        types::for_each(
            types::forward_iterator_list<MovableNotCopiable*>{},
            TestIteratorWithPolicies<types::partial_instantiation<TestMovableNotCopiable, Iter>::template apply>{});
      }});
  return 0;
}
