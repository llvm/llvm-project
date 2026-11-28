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
//           class ForwardIterator2,
//           class T>
//   ForwardIterator2 remove_copy(ExecutionPolicy&& exec,
//                                ForwardIterator1 first,
//                                ForwardIterator1 last,
//                                ForwardIterator2 result,
//                                const T& value);

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

EXECUTION_POLICY_SFINAE_TEST(remove_copy);

static_assert(sfinae_test_remove_copy<int, int*, int*, int*, int>);
static_assert(!sfinae_test_remove_copy<std::execution::parallel_policy, int*, int*, int*, int>);

// A type which provides only operator==, to check that remove_copy doesn't require operator!=.
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
      auto r              = std::remove_copy(policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(o)), 1);
      static_assert(std::is_same_v<decltype(r), Iter2>);
    }
    { // Check empty
      int a[] = {1};
      int o[] = {-1};
      auto r  = std::remove_copy(policy, Iter1(std::begin(a)), Iter1(std::begin(a)), Iter2(std::begin(o)), 1);
      assert(r == Iter2(std::begin(o)));
      assert(o[0] == -1);
    }
    { // Check one removed
      int a[] = {1};
      int o[] = {-1};
      auto r  = std::remove_copy(policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(o)), 1);
      assert(r == Iter2(std::begin(o)));
      assert(o[0] == -1);
    }
    { // Check one kept
      int a[] = {2};
      int o[] = {-1};
      auto r  = std::remove_copy(policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(o)), 1);
      assert(r == Iter2(std::end(o)));
      assert(o[0] == 2);
    }
    { // Check two, removed then kept
      int a[] = {1, 2};
      int o[] = {-1, -1};
      auto r  = std::remove_copy(policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(o)), 1);
      assert(r == Iter2(std::begin(o) + 1));
      assert(o[0] == 2);
      assert(o[1] == -1);
    }
    { // Check two, kept then removed
      int a[] = {2, 1};
      int o[] = {-1, -1};
      auto r  = std::remove_copy(policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(o)), 1);
      assert(r == Iter2(std::begin(o) + 1));
      assert(o[0] == 2);
      assert(o[1] == -1);
    }
    { // Check mixed, the elements past the returned iterator are untouched
      int a[]             = {3, 1, 3, 4, 3, 3, 5, 6, 3};
      int o[std::size(a)] = {-1, -1, -1, -1, -1, -1, -1, -1, -1};
      auto r              = std::remove_copy(policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(o)), 3);
      assert(r == Iter2(std::begin(o) + 4));
      assert(o[0] == 1);
      assert(o[1] == 4);
      assert(o[2] == 5);
      assert(o[3] == 6);
      assert(std::all_of(std::begin(o) + 4, std::end(o), [](int v) { return v == -1; }));
    }
    { // Check all removed
      int a[10];
      std::fill(std::begin(a), std::end(a), 7);
      int o[std::size(a)];
      std::fill(std::begin(o), std::end(o), -1);
      auto r = std::remove_copy(policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(o)), 7);
      assert(r == Iter2(std::begin(o)));
      assert(std::all_of(std::begin(o), std::end(o), [](int v) { return v == -1; }));
    }
    { // Check all kept
      int a[10];
      std::iota(std::begin(a), std::end(a), 0);
      int o[std::size(a)] = {0};
      auto r = std::remove_copy(policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(o)), 42);
      assert(r == Iter2(std::end(o)));
      assert(std::equal(std::begin(a), std::end(a), std::begin(o)));
    }
    { // Check all removed but first
      int a[10];
      std::fill(std::begin(a), std::end(a), 7);
      a[0]                = 10;
      int o[std::size(a)] = {0};
      auto r              = std::remove_copy(policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(o)), 7);
      assert(r == Iter2(std::begin(o) + 1));
      assert(o[0] == 10);
    }
    { // Check all removed but last
      int a[10];
      std::fill(std::begin(a), std::end(a), 7);
      a[std::size(a) - 1] = 10;
      int o[std::size(a)] = {0};
      auto r              = std::remove_copy(policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(o)), 7);
      assert(r == Iter2(std::begin(o) + 1));
      assert(o[0] == 10);
    }
    { // Check iota'ed input modulo 3 sampled at different lengths, value: 1
      int a[1073];
      int o[std::size(a)];
      std::generate(std::begin(a), std::end(a), [v = 0]() mutable { return v++ % 3; });
      runway_sample(std::size(a) + 1, [&](std::size_t n) {
        std::fill(std::begin(o), std::end(o), -1);
        auto r = std::remove_copy(policy, Iter1(std::begin(a)), Iter1(std::begin(a) + n), Iter2(std::begin(o)), 1);
        std::size_t count = n - (n + 1) / 3;
        assert(r == Iter2(std::begin(o) + count));
        for (std::size_t i = 0; i != std::size(o); ++i)
          assert(o[i] == (i < count ? (i % 2 == 0 ? 0 : 2) : -1));
      });
    }
  }
};

template <class Iter1, class Iter2>
struct TestEqualityOnly {
  template <class ExecutionPolicy>
  void operator()(ExecutionPolicy&& policy) {
    EqualityOnly a[] = {{1}, {2}, {1}, {3}, {1}};
    EqualityOnly o[] = {{-1}, {-1}, {-1}, {-1}, {-1}};
    auto r = std::remove_copy(policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(o)), EqualityOnly{1});
    assert(r == Iter2(std::begin(o) + 2));
    assert(o[0].v == 2);
    assert(o[1].v == 3);
    assert(o[2].v == -1);
    assert(o[3].v == -1);
    assert(o[4].v == -1);
  }
};

template <class Iter1, class Iter2>
struct TestString {
  template <class ExecutionPolicy>
  void operator()(ExecutionPolicy&& policy) {
    std::string value = "c";
    for (std::size_t size : {0, 1, 2, 3, 10, 100}) {
      std::vector<std::string> data(size);
      for (std::size_t i = 0; i < size; ++i)
        data[i] = std::string(1, static_cast<char>('a' + i % 4)); // a, b, c, a, b, c, ...

      std::vector<std::string> result(size);
      auto ret =
          std::remove_copy(policy, Iter1(data.data()), Iter1(data.data() + data.size()), Iter2(result.data()), value);

      std::vector<std::string> expected;
      std::remove_copy(data.begin(), data.end(), std::back_inserter(expected), value);
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
