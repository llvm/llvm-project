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
//   ForwardIterator2 move(ExecutionPolicy&& policy,
//                         ForwardIterator1 first, ForwardIterator1 last,
//                         ForwardIterator2 result);

#include <algorithm>
#include <vector>

#include "test_macros.h"
#include "test_execution_policies.h"
#include "test_iterators.h"

EXECUTION_POLICY_SFINAE_TEST(move);

static_assert(sfinae_test_move<int, int*, int*, int*>);
static_assert(!sfinae_test_move<std::execution::parallel_policy, int*, int*, int*>);

template <class Iter1, class Iter2>
struct TestInt {
  template <class Policy>
  void operator()(Policy&& policy) {
    // simple test
    for (const int size : {0, 1, 2, 100, 350}) {
      std::vector<int> a(size);
      for (int i = 0; i != size; ++i)
        a[i] = i + 1;

      std::vector<int> out(std::size(a));
      decltype(auto) ret =
          std::move(policy, Iter1(std::data(a)), Iter1(std::data(a) + std::size(a)), Iter2(std::data(out)));
      static_assert(std::is_same_v<decltype(ret), Iter2>);
      assert(base(ret) == std::data(out) + std::size(out));
      for (int i = 0; i != size; ++i)
        assert(out[i] == i + 1);
    }
  }
};

struct MovedToTester {
  bool moved_to   = false;
  MovedToTester() = default;
  MovedToTester(MovedToTester&&) {}
  MovedToTester& operator=(MovedToTester&&) {
    assert(!moved_to);
    moved_to = true;
    return *this;
  }
  ~MovedToTester() = default;
};

template <class Iter1, class Iter2>
struct TestNonTrivial {
  template <class Policy>
  void operator()(Policy&& policy) {
    // simple test
    for (const int size : {0, 1, 2, 100, 350}) {
      std::vector<MovedToTester> a(size);

      std::vector<MovedToTester> out(std::size(a));
      auto ret = std::move(policy, Iter1(std::data(a)), Iter1(std::data(a) + std::size(a)), Iter2(std::data(out)));
      assert(base(ret) == std::data(out) + std::size(out));
      assert(std::all_of(std::begin(out), std::end(out), [](MovedToTester& t) { return t.moved_to; }));
      assert(std::none_of(std::begin(a), std::end(a), [](MovedToTester& t) { return t.moved_to; }));
    }
  }
};

int main(int, char**) {
  types::for_each(types::forward_iterator_list<int*>{}, types::apply_type_identity{[](auto v) {
                    using Iter = typename decltype(v)::type;
                    types::for_each(
                        types::forward_iterator_list<int*>{},
                        TestIteratorWithPolicies< types::partial_instantiation<TestInt, Iter>::template apply>{});
                  }});

  types::for_each(
      types::forward_iterator_list<MovedToTester*>{}, types::apply_type_identity{[](auto v) {
        using Iter = typename decltype(v)::type;
        types::for_each(
            types::forward_iterator_list<MovedToTester*>{},
            TestIteratorWithPolicies< types::partial_instantiation<TestNonTrivial, Iter>::template apply>{});
      }});

  return 0;
}
