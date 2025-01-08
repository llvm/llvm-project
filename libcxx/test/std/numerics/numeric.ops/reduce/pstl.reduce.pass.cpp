//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// UNSUPPORTED: libcpp-has-no-incomplete-pstl

// <numeric>

// template<class ExecutionPolicy, class ForwardIterator>
//   typename iterator_traits<ForwardIterator>::value_type
//     reduce(ExecutionPolicy&& exec,
//            ForwardIterator first, ForwardIterator last);
// template<class ExecutionPolicy, class ForwardIterator, class T, class BinaryOperation>
//   T reduce(ExecutionPolicy&& exec,
//            ForwardIterator first, ForwardIterator last, T init,
//            BinaryOperation binary_op);

#include <numeric>
#include <vector>

#include "MoveOnly.h"
#include "test_execution_policies.h"
#include "test_iterators.h"
#include "test_macros.h"

template <class Iter, class ValueT>
struct Test {
  template <class Policy>
  void operator()(Policy&& policy) {
    for (const auto& pair : {std::pair{0, 34}, {1, 36}, {2, 39}, {100, 5184}, {350, 61809}}) {
      auto [size, expected] = pair;
      std::vector<int> a(size);
      for (int i = 0; i != size; ++i)
        a[i] = i;

      {
        decltype(auto) ret = std::reduce(
            policy, Iter(std::data(a)), Iter(std::data(a) + std::size(a)), ValueT(34), [](ValueT i, ValueT j) {
              return i + j + 2;
            });
        static_assert(std::is_same_v<decltype(ret), ValueT>);
        assert(ret == ValueT(expected));
      }
      {
        decltype(auto) ret = std::reduce(policy, Iter(std::data(a)), Iter(std::data(a) + std::size(a)), ValueT(34));
        static_assert(std::is_same_v<decltype(ret), ValueT>);
        assert(ret == expected - 2 * size);
      }
      {
        decltype(auto) ret = std::reduce(policy, Iter(std::data(a)), Iter(std::data(a) + std::size(a)));
        static_assert(std::is_same_v<decltype(ret), typename std::iterator_traits<Iter>::value_type>);
        assert(ret == expected - 2 * size - 34);
      }
    }
  }
};

int main(int, char**) {
  types::for_each(types::forward_iterator_list<int*>{}, types::apply_type_identity{[](auto v) {
                    using Iter = typename decltype(v)::type;
                    types::for_each(
                        types::type_list<int, MoveOnly>{},
                        TestIteratorWithPolicies<types::partial_instantiation<Test, Iter>::template apply>{});
                  }});

  return 0;
}
