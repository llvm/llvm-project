//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// mapped_type& operator[](const key_type& k);

#include <cassert>
#include <deque>
#include <flat_map>
#include <functional>
#include <vector>

#include "MinSequenceContainer.h"
#include "../helpers.h"
#include "min_allocator.h"
#include "test_macros.h"

// Constraints: is_constructible_v<mapped_type> is true.
template <class M, class Input>
concept CanIndex = requires(M m, Input k) { m[k]; };

static_assert(CanIndex<std::flat_map<int, double>, const int&>);
static_assert(!CanIndex<std::flat_map<int, NoDefaultCtr>, const int&>);

template <class KeyContainer, class ValueContainer>
void test() {
  using P = std::pair<int, double>;
  P ar[]  = {
      P(1, 1.5),
      P(2, 2.5),
      P(3, 3.5),
      P(4, 4.5),
      P(5, 5.5),
      P(7, 7.5),
      P(8, 8.5),
  };
  const int one = 1;
  std::flat_map<int, double, std::less<int>, KeyContainer, ValueContainer> m(ar, ar + sizeof(ar) / sizeof(ar[0]));
  ASSERT_SAME_TYPE(decltype(m[one]), double&);
  assert(m.size() == 7);
  assert(m[one] == 1.5);
  assert(m.size() == 7);
  m[1] = -1.5;
  assert(m[1] == -1.5);
  assert(m.size() == 7);
  assert(m[6] == 0);
  assert(m.size() == 8);
  m[6] = 6.5;
  assert(m[6] == 6.5);
  assert(m.size() == 8);
}

int main(int, char**) {
  test<std::vector<int>, std::vector<double>>();
  test<std::deque<int>, std::vector<double>>();
  test<MinSequenceContainer<int>, MinSequenceContainer<double>>();
  test<std::vector<int, min_allocator<int>>, std::vector<double, min_allocator<double>>>();

  {
    auto index_func = [](auto& m, auto key_arg, auto value_arg) {
      using FlatMap                             = std::decay_t<decltype(m)>;
      const typename FlatMap::key_type key      = key_arg;
      const typename FlatMap::mapped_type value = value_arg;
      m[key]                                    = value;
    };
    test_emplace_exception_guarantee(index_func);
  }
  return 0;
}
