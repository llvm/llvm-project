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
//          class UnaryOperation>
//   ForwardIterator2
//     transform(ExecutionPolicy&& exec,
//               ForwardIterator1 first1, ForwardIterator1 last1,
//               ForwardIterator2 result, UnaryOperation op);

#include <algorithm>
#include <cassert>
#include <vector>

#include "test_macros.h"
#include "test_execution_policies.h"
#include "test_iterators.h"

// We can't test the constraint on the execution policy, because that would conflict with the binary
// transform algorithm that doesn't take an execution policy, which is not constrained at all.

template <class Iter1, class Iter2>
struct Test {
  template <class Policy>
  void operator()(Policy&& policy) {
    // simple test
    for (const int size : {0, 1, 2, 100, 350}) {
      std::vector<int> a(size);
      for (int i = 0; i != size; ++i)
        a[i] = i + 1;

      std::vector<int> out(std::size(a));
      decltype(auto) ret = std::transform(
          policy, Iter1(std::data(a)), Iter1(std::data(a) + std::size(a)), Iter2(std::data(out)), [](int i) {
            return i + 3;
          });
      static_assert(std::is_same_v<decltype(ret), Iter2>);
      assert(base(ret) == std::data(out) + std::size(out));
      for (int i = 0; i != size; ++i)
        assert(out[i] == i + 4);
    }
  }
};

int main(int, char**) {
  types::for_each(types::forward_iterator_list<int*>{}, types::apply_type_identity{[](auto v) {
                    using Iter = typename decltype(v)::type;
                    types::for_each(
                        types::forward_iterator_list<int*>{},
                        TestIteratorWithPolicies<types::partial_instantiation<Test, Iter>::template apply>{});
                  }});

  return 0;
}
