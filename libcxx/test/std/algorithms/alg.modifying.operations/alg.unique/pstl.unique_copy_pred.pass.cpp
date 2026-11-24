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
//           class BinaryPredicate>
//   ForwardIterator2 unique_copy(ExecutionPolicy&& exec,
//                                ForwardIterator1 first,
//                                ForwardIterator1 last,
//                                ForwardIterator2 result,
//                                BinaryPredicate pred);

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

static_assert(sfinae_test_unique_copy<int, int*, int*, int*, bool (*)(int, int)>);
static_assert(!sfinae_test_unique_copy<std::execution::parallel_policy, int*, int*, int*, bool (*)(int, int)>);

// A type without any comparison operators.
struct NotEqualityComparable {
  int v;
};

template <class Iter1, class Iter2>
struct TestInt {
  template <class ExecutionPolicy>
  void operator()(ExecutionPolicy&& policy) {
    // Considers two values equivalent if they belong to the same decade.
    auto comp = [](int x, int y) { return x / 10 == y / 10; };
    { // Check the return type
      int a[]             = {0};
      int o[std::size(a)] = {0};
      auto r = std::unique_copy(policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(o)), comp);
      static_assert(std::is_same_v<decltype(r), Iter2>);
    }
    { // Check empty
      int a[] = {1};
      int o[] = {-1};
      auto r  = std::unique_copy(policy, Iter1(std::begin(a)), Iter1(std::begin(a)), Iter2(std::begin(o)), comp);
      assert(r == Iter2(std::begin(o)));
      assert(o[0] == -1);
    }
    { // Check one element
      int a[] = {1};
      int o[] = {-1};
      auto r  = std::unique_copy(policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(o)), comp);
      assert(r == Iter2(std::end(o)));
      assert(o[0] == 1);
    }
    { // Check two, equivalent, the first one is kept
      int a[] = {11, 12};
      int o[] = {-1, -1};
      auto r  = std::unique_copy(policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(o)), comp);
      assert(r == Iter2(std::begin(o) + 1));
      assert(o[0] == 11);
      assert(o[1] == -1);
    }
    { // Check two, not equivalent
      int a[] = {11, 21};
      int o[] = {-1, -1};
      auto r  = std::unique_copy(policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(o)), comp);
      assert(r == Iter2(std::end(o)));
      assert(o[0] == 11);
      assert(o[1] == 21);
    }
    { // Check mixed
      int a[]             = {10, 11, 12, 20, 25, 13, 19, 30, 40};
      int o[std::size(a)] = {-1, -1, -1, -1, -1, -1, -1, -1, -1};
      auto r = std::unique_copy(policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(o)), comp);
      assert(r == Iter2(std::begin(o) + 5));
      assert(o[0] == 10);
      assert(o[1] == 20);
      assert(o[2] == 13);
      assert(o[3] == 30);
      assert(o[4] == 40);
      assert(std::all_of(std::begin(o) + 5, std::end(o), [](int v) { return v == -1; }));
    }
    { // Check checkerboard
      int a[]             = {1, 21, 2, 22, 3};
      int o[std::size(a)] = {-1, -1, -1, -1, -1};
      auto r = std::unique_copy(policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(o)), comp);
      assert(r == Iter2(std::end(o)));
      assert(std::equal(std::begin(a), std::end(a), std::begin(o)));
    }
    { // Check all equivalent
      int a[10];
      std::iota(std::begin(a), std::end(a), 70);
      int o[std::size(a)];
      std::fill(std::begin(o), std::end(o), -1);
      auto r = std::unique_copy(policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(o)), comp);
      assert(r == Iter2(std::begin(o) + 1));
      assert(o[0] == 70);
      assert(std::all_of(std::begin(o) + 1, std::end(o), [](int v) { return v == -1; }));
    }
    { // Check none equivalent
      int a[10];
      std::generate(std::begin(a), std::end(a), [v = 0]() mutable { return 10 * v++; });
      int o[std::size(a)] = {0};
      auto r = std::unique_copy(policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(o)), comp);
      assert(r == Iter2(std::end(o)));
      assert(std::equal(std::begin(a), std::end(a), std::begin(o)));
    }
    { // Check the order of arguments passed to the predicate
      // The predicate is called with the earlier element as the first argument. The input is strictly increasing,
      // so the first argument must be smaller than the second.
      auto check_order = [](int x, int y) {
        assert(x < y);
        return x / 2 == y / 2;
      };
      int a[10];
      std::iota(std::begin(a), std::end(a), 0);
      int o[std::size(a)] = {0};
      auto r = std::unique_copy(policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(o)), check_order);
      assert(r == Iter2(std::begin(o) + 5));
      for (std::size_t i = 0; i != 5; ++i)
        assert(o[i] == static_cast<int>(2 * i));
    }
    { // Check iota'ed input sampled at different lengths, pred: x / 3 == y / 3
      int a[1073];
      int o[std::size(a)];
      std::iota(std::begin(a), std::end(a), 0);
      runway_sample(std::size(a) + 1, [&](std::size_t n) {
        std::fill(std::begin(o), std::end(o), -1);
        auto r = std::unique_copy(
            policy, Iter1(std::begin(a)), Iter1(std::begin(a) + n), Iter2(std::begin(o)), [](int x, int y) {
              return x / 3 == y / 3;
            });
        std::size_t count = (n + 2) / 3;
        assert(r == Iter2(std::begin(o) + count));
        for (std::size_t i = 0; i < std::size(o); ++i)
          assert(o[i] == (i < count ? static_cast<int>(3 * i) : -1));
      });
    }
  }
};

template <class Iter1, class Iter2>
struct TestNotEqualityComparable {
  template <class ExecutionPolicy>
  void operator()(ExecutionPolicy&& policy) {
    // Considers two values equivalent if they belong to the same decade.
    auto comp = [](const NotEqualityComparable& x, const NotEqualityComparable& y) { return x.v / 10 == y.v / 10; };
    NotEqualityComparable a[] = {{1}, {2}, {11}, {3}, {31}, {32}};
    NotEqualityComparable o[] = {{-1}, {-1}, {-1}, {-1}, {-1}, {-1}};
    auto r = std::unique_copy(policy, Iter1(std::begin(a)), Iter1(std::end(a)), Iter2(std::begin(o)), comp);
    assert(r == Iter2(std::begin(o) + 4));
    assert(o[0].v == 1);
    assert(o[1].v == 11);
    assert(o[2].v == 3);
    assert(o[3].v == 31);
    assert(o[4].v == -1);
    assert(o[5].v == -1);
  }
};

template <class Iter1, class Iter2>
struct TestString {
  template <class ExecutionPolicy>
  void operator()(ExecutionPolicy&& policy) {
    auto pred = [](const std::string& x, const std::string& y) { return x == y; };
    for (std::size_t size : {0, 1, 2, 3, 10, 100}) {
      std::vector<std::string> data(size);
      for (std::size_t i = 0; i < size; ++i)
        data[i] = std::string(1, static_cast<char>('a' + (i / 3) % 3)); // a, a, a, b, b, b, c, c, c, a, a, a, ...

      std::vector<std::string> result(size);
      auto ret =
          std::unique_copy(policy, Iter1(data.data()), Iter1(data.data() + data.size()), Iter2(result.data()), pred);

      std::size_t count = (size + 2) / 3;
      assert(ret == Iter2(result.data() + count));
      for (std::size_t i = 0; i < size; ++i)
        assert(result[i] == (i < count ? data[3 * i] : std::string()));
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
      types::forward_iterator_list<const NotEqualityComparable*>{}, types::apply_type_identity{[](auto v) {
        using Iter = typename decltype(v)::type;
        types::for_each(
            types::forward_iterator_list<NotEqualityComparable*>{},
            TestIteratorWithPolicies<types::partial_instantiation<TestNotEqualityComparable, Iter>::template apply>{});
      }});
  types::for_each(types::forward_iterator_list<const std::string*>{}, types::apply_type_identity{[](auto v) {
                    using Iter = typename decltype(v)::type;
                    types::for_each(
                        types::forward_iterator_list<std::string*>{},
                        TestIteratorWithPolicies<types::partial_instantiation<TestString, Iter>::template apply>{});
                  }});
  return 0;
}
