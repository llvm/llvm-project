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

// template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2,
//          class ForwardIterator, class BinaryOperation>
//   ForwardIterator
//     transform(ExecutionPolicy&& exec,
//               ForwardIterator1 first1, ForwardIterator1 last1,
//               ForwardIterator2 first2, ForwardIterator result,
//               BinaryOperation binary_op);

#include <algorithm>
#include <vector>

#include "test_macros.h"
#include "test_execution_policies.h"
#include "test_iterators.h"

EXECUTION_POLICY_SFINAE_TEST(transform);

static_assert(sfinae_test_transform<int, int*, int*, int*, int*, bool (*)(int)>);
static_assert(!sfinae_test_transform<std::execution::parallel_policy, int*, int*, int*, int*, int (*)(int, int)>);

template <class Iter1, class Iter2, class Iter3>
struct Test {
  template <class Policy>
  void operator()(Policy&& policy) {
    // simple test
    for (const int size : {0, 1, 2, 100, 350}) {
      std::vector<int> a(size);
      std::vector<int> b(size);
      for (int i = 0; i != size; ++i) {
        a[i] = i + 1;
        b[i] = i - 3;
      }

      std::vector<int> out(std::size(a));
      decltype(auto) ret = std::transform(
          policy,
          Iter1(std::data(a)),
          Iter1(std::data(a) + std::size(a)),
          Iter2(std::data(b)),
          Iter3(std::data(out)),
          [](int i, int j) { return i + j + 3; });
      static_assert(std::is_same_v<decltype(ret), Iter3>);
      assert(base(ret) == std::data(out) + std::size(out));
      for (int i = 0; i != size; ++i) {
        assert(out[i] == i * 2 + 1);
      }
    }
  }
};

int main(int, char**) {
  types::for_each(
      types::forward_iterator_list<int*>{}, types::apply_type_identity{[](auto v) {
        using Iter3 = typename decltype(v)::type;
        types::for_each(
            types::forward_iterator_list<int*>{}, types::apply_type_identity{[](auto v2) {
              using Iter2 = typename decltype(v2)::type;
              types::for_each(
                  types::forward_iterator_list<int*>{},
                  TestIteratorWithPolicies<types::partial_instantiation<Test, Iter2, Iter3>::template apply>{});
            }});
      }});

  return 0;
}
