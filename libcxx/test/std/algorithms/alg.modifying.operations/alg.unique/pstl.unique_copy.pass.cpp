//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// REQUIRES: std-at-least-c++17

// UNSUPPORTED: libcpp-has-no-incomplete-pstl

// template <class ExecutionPolicy,
//           class ForwardIterator1,
//           class ForwardIterator2>
//   ForwardIterator2 unique_copy(ExecutionPolicy&& exec,
//                                ForwardIterator1 first,
//                                ForwardIterator1 last,
//                                ForwardIterator2 result);

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iterator>
#include <numeric>
#include <string>
#include <type_traits>
#include <vector>

#include "test_execution_policies.h"
#include "test_iterators.h"
#include "runway_sample.h"
#include "test_macros.h"
#include "type_algorithms.h"

EXECUTION_POLICY_SFINAE_TEST(unique_copy);

static_assert(sfinae_test_unique_copy<int, int*, int*, int*>);
static_assert(!sfinae_test_unique_copy<std::execution::parallel_policy, int*, int*, int*>);

// A type which provides only operator== (and no operator!=).
struct EqualityOnly {
  int v;
  friend bool operator==(const EqualityOnly& lhs, const EqualityOnly& rhs) { return lhs.v == rhs.v; }
};

template <class Iter1, class Iter2>
struct TestInt {
  template <class ExecutionPolicy>
  void operator()(ExecutionPolicy&& policy) {
    { // Check the return type
      int a[]             = {0};
      int o[std::size(a)] = {0};
      auto r              = std::unique_copy(policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(o)));
      static_assert(std::is_same_v<decltype(r), Iter2>);
    }
    { // Check empty
      int a[] = {1};
      int o[] = {-1};
      auto r  = std::unique_copy(policy, Iter1(std::begin(a)), Iter1(std::begin(a)), Iter2(std::begin(o)));
      assert(r == Iter2(std::begin(o)));
      assert(o[0] == -1);
    }
    { // Check one element
      int a[] = {1};
      int o[] = {-1};
      auto r  = std::unique_copy(policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(o)));
      assert(r == Iter2(std::end(o)));
      assert(o[0] == 1);
    }
    { // Check two, equal
      int a[] = {1, 1};
      int o[] = {-1, -1};
      auto r  = std::unique_copy(policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(o)));
      assert(r == Iter2(std::begin(o) + 1));
      assert(o[0] == 1);
      assert(o[1] == -1);
    }
    { // Check two, different
      int a[] = {1, 2};
      int o[] = {-1, -1};
      auto r  = std::unique_copy(policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(o)));
      assert(r == Iter2(std::end(o)));
      assert(o[0] == 1);
      assert(o[1] == 2);
    }
    { // Check mixed
      int a[]             = {1, 1, 2, 2, 2, 3, 1, 1, 4};
      int o[std::size(a)] = {-1, -1, -1, -1, -1, -1, -1, -1, -1};
      auto r              = std::unique_copy(policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(o)));
      assert(r == Iter2(std::begin(o) + 5));
      assert(o[0] == 1);
      assert(o[1] == 2);
      assert(o[2] == 3);
      assert(o[3] == 1);
      assert(o[4] == 4);
      assert(std::all_of(std::begin(o) + 5, std::end(o), [](int v) { return v == -1; }));
    }
    { // Check checkerboard
      int a[]             = {1, 2, 1, 2, 1};
      int o[std::size(a)] = {-1, -1, -1, -1, -1};
      auto r              = std::unique_copy(policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(o)));
      assert(r == Iter2(std::end(o)));
      assert(std::equal(std::begin(a), std::end(a), std::begin(o)));
    }
    { // Check all equal
      int a[10];
      std::fill(std::begin(a), std::end(a), 7);
      int o[std::size(a)];
      std::fill(std::begin(o), std::end(o), -1);
      auto r = std::unique_copy(policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(o)));
      assert(r == Iter2(std::begin(o) + 1));
      assert(o[0] == 7);
      assert(std::all_of(std::begin(o) + 1, std::end(o), [](int v) { return v == -1; }));
    }
    { // Check all different
      int a[10];
      std::iota(std::begin(a), std::end(a), 0);
      int o[std::size(a)] = {0};
      auto r              = std::unique_copy(policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(o)));
      assert(r == Iter2(std::end(o)));
      assert(std::equal(std::begin(a), std::end(a), std::begin(o)));
    }
    { // Check all equal but first
      int a[10];
      std::fill(std::begin(a), std::end(a), 7);
      a[0]                = 10;
      int o[std::size(a)] = {0};
      auto r              = std::unique_copy(policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(o)));
      assert(r == Iter2(std::begin(o) + 2));
      assert(o[0] == 10);
      assert(o[1] == 7);
    }
    { // Check all equal but last
      int a[10];
      std::fill(std::begin(a), std::end(a), 7);
      a[std::size(a) - 1] = 10;
      int o[std::size(a)] = {0};
      auto r              = std::unique_copy(policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(o)));
      assert(r == Iter2(std::begin(o) + 2));
      assert(o[0] == 7);
      assert(o[1] == 10);
    }
    { // Check input of groups of 3 equal elements sampled at different lengths
      int a[1073];
      int o[std::size(a)];
      std::generate(std::begin(a), std::end(a), [v = 0]() mutable { return v++ / 3; });
      runway_sample(std::size(a) + 1, [&](std::size_t n) {
        std::fill(std::begin(o), std::end(o), -1);
        auto r = std::unique_copy(policy, Iter1(std::begin(a)), Iter1(std::begin(a) + n), Iter2(std::begin(o)));
        std::size_t count = (n + 2) / 3;
        assert(r == Iter2(std::begin(o) + count));
        for (std::size_t i = 0; i < std::size(o); ++i)
          assert(o[i] == (i < count ? static_cast<int>(i) : -1));
      });
    }
  }
};

template <class Iter1, class Iter2>
struct TestEqualityOnly {
  template <class ExecutionPolicy>
  void operator()(ExecutionPolicy&& policy) {
    EqualityOnly a[] = {{1}, {1}, {2}, {1}, {3}, {3}};
    EqualityOnly o[] = {{-1}, {-1}, {-1}, {-1}, {-1}, {-1}};
    auto r           = std::unique_copy(policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(o)));
    assert(r == Iter2(std::begin(o) + 4));
    assert(o[0].v == 1);
    assert(o[1].v == 2);
    assert(o[2].v == 1);
    assert(o[3].v == 3);
    assert(o[4].v == -1);
    assert(o[5].v == -1);
  }
};

template <class Iter1, class Iter2>
struct TestString {
  template <class ExecutionPolicy>
  void operator()(ExecutionPolicy&& policy) {
    for (std::size_t size : {0, 1, 2, 3, 10, 100}) {
      std::vector<std::string> data(size);
      for (std::size_t i = 0; i < size; ++i)
        data[i] = std::string(1, static_cast<char>('a' + (i / 3) % 3)); // a, a, a, b, b, b, c, c, c, a, a, a, ...

      std::vector<std::string> result(size);
      auto ret = std::unique_copy(policy, Iter1(data.data()), Iter1(data.data() + data.size()), Iter2(result.data()));

      std::vector<std::string> expected;
      std::unique_copy(data.begin(), data.end(), std::back_inserter(expected));
      assert(ret == Iter2(result.data() + expected.size()));
      assert(std::equal(expected.begin(), expected.end(), result.begin()));
      assert(std::all_of(ret, Iter2(result.data() + result.size()), [](auto& s) { return s.empty(); }));
    }
  }
};

int main(int, char**) {
  types::for_each(types::forward_iterator_list<const int*>{}, types::apply_type_identity{[](auto v) {
                    using Iter = typename decltype(v)::type;
                    types::for_each(
                        types::forward_iterator_list<int*>{},
                        TestIteratorWithPolicies<types::partial_instantiation<TestInt, Iter>::template apply>{});
                  }});
  types::for_each(
      types::forward_iterator_list<const EqualityOnly*>{}, types::apply_type_identity{[](auto v) {
        using Iter = typename decltype(v)::type;
        types::for_each(
            types::forward_iterator_list<EqualityOnly*>{},
            TestIteratorWithPolicies<types::partial_instantiation<TestEqualityOnly, Iter>::template apply>{});
      }});
  types::for_each(types::forward_iterator_list<const std::string*>{}, types::apply_type_identity{[](auto v) {
                    using Iter = typename decltype(v)::type;
                    types::for_each(
                        types::forward_iterator_list<std::string*>{},
                        TestIteratorWithPolicies<types::partial_instantiation<TestString, Iter>::template apply>{});
                  }});
  return 0;
}
