//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// REQUIRES: with-pstl

// <algorithm>

// template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2>
//   ForwardIterator2 copy(ExecutionPolicy&& policy,
//                         ForwardIterator1 first, ForwardIterator1 last,
//                         ForwardIterator2 result);

#include <algorithm>
#include <vector>

#include "test_macros.h"
#include "test_execution_policies.h"
#include "test_iterators.h"

EXECUTION_POLICY_SFINAE_TEST(copy);

static_assert(sfinae_test_copy<int, int*, int*, bool (*)(int)>);
static_assert(!sfinae_test_copy<std::execution::parallel_policy, int*, int*, int>);

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
          std::copy(policy, Iter1(std::data(a)), Iter1(std::data(a) + std::size(a)), Iter2(std::data(out)));
      static_assert(std::is_same_v<decltype(ret), Iter2>);
      assert(base(ret) == std::data(out) + std::size(out));
      for (int i = 0; i != size; ++i)
        assert(out[i] == i + 1);
    }
  }
};

struct CopiedToTester {
  bool copied_to   = false;
  CopiedToTester() = default;
  CopiedToTester(const CopiedToTester&) {}
  CopiedToTester& operator=(const CopiedToTester&) {
    assert(!copied_to);
    copied_to = true;
    return *this;
  }
  ~CopiedToTester() = default;
};

template <class Iter1, class Iter2>
struct TestNonTrivial {
  template <class Policy>
  void operator()(Policy&& policy) {
    // simple test
    for (const int size : {0, 1, 2, 100, 350}) {
      std::vector<CopiedToTester> a(size);

      std::vector<CopiedToTester> out(std::size(a));
      auto ret = std::copy(policy, Iter1(std::data(a)), Iter1(std::data(a) + std::size(a)), Iter2(std::data(out)));
      assert(base(ret) == std::data(out) + std::size(out));
      assert(std::all_of(std::begin(out), std::end(out), [](CopiedToTester& t) { return t.copied_to; }));
      assert(std::none_of(std::begin(a), std::end(a), [](CopiedToTester& t) { return t.copied_to; }));
    }
  }
};

struct TestIteratorsNonTrivial {
  template <class Iter2>
  void operator()() {}
};

int main(int, char**) {
  types::for_each(types::forward_iterator_list<int*>{}, types::apply_type_identity{[](auto v) {
                    using Iter = typename decltype(v)::type;
                    types::for_each(
                        types::forward_iterator_list<int*>{},
                        TestIteratorWithPolicies< types::partial_instantiation<TestInt, Iter>::template apply>{});
                  }});

  types::for_each(
      types::forward_iterator_list<CopiedToTester*>{}, types::apply_type_identity{[](auto v) {
        using Iter = typename decltype(v)::type;
        types::for_each(
            types::forward_iterator_list<CopiedToTester*>{},
            TestIteratorWithPolicies< types::partial_instantiation<TestNonTrivial, Iter>::template apply>{});
      }});

  return 0;
}
