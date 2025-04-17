//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// template<class K> mapped_type& operator[](K&& x);

#include <cassert>
#include <deque>
#include <flat_map>
#include <functional>
#include <vector>

#include "MinSequenceContainer.h"
#include "../helpers.h"
#include "test_macros.h"
#include "min_allocator.h"

// Constraints:
// The qualified-id Compare::is_transparent is valid and denotes a type.
// is_constructible_v<key_type, K> is true.
// is_constructible_v<mapped_type, Args...> is true.
// is_convertible_v<K&&, const_iterator> and is_convertible_v<K&&, iterator> are both false
template <class M, class Input>
concept CanIndex                      = requires(M m, Input k) { m[k]; };
using TransparentMap                  = std::flat_map<int, double, TransparentComparator>;
using NonTransparentMap               = std::flat_map<int, double, NonTransparentComparator>;
using TransparentNoDefaultCtrValueMap = std::flat_map<int, NoDefaultCtr, TransparentComparator>;

static_assert(CanIndex<TransparentMap, ConvertibleTransparent<int>>);
static_assert(!CanIndex<const TransparentMap, ConvertibleTransparent<int>>);

static_assert(!CanIndex<NonTransparentMap, NonConvertibleTransparent<int>>);
static_assert(!CanIndex<const NonTransparentMap, NonConvertibleTransparent<int>>);

static_assert(!CanIndex<TransparentMap, NonConvertibleTransparent<int>>);
static_assert(!CanIndex<const TransparentMap, NonConvertibleTransparent<int>>);

static_assert(!CanIndex<TransparentNoDefaultCtrValueMap, ConvertibleTransparent<int>>);
static_assert(!CanIndex<const TransparentNoDefaultCtrValueMap, ConvertibleTransparent<int>>);

static_assert(!CanIndex<TransparentMap, TransparentMap::iterator>);
static_assert(!CanIndex<TransparentMap, TransparentMap::const_iterator>);

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
  const ConvertibleTransparent<int> one{1};
  const ConvertibleTransparent<int> six{6};
  {
    std::flat_map<int, double, TransparentComparator, KeyContainer, ValueContainer> m(
        ar, ar + sizeof(ar) / sizeof(ar[0]));
    ASSERT_SAME_TYPE(decltype(m[one]), double&);
    assert(m.size() == 7);
    assert(m[one] == 1.5);
    assert(m.size() == 7);
    m[one] = -1.5;
    assert(m[one] == -1.5);
    assert(m.size() == 7);
    assert(m[six] == 0);
    assert(m.size() == 8);
    m[six] = 6.5;
    assert(m[six] == 6.5);
    assert(m.size() == 8);
  }
}

int main(int, char**) {
  test<std::vector<int>, std::vector<double>>();
  test<std::deque<int>, std::vector<double>>();
  test<MinSequenceContainer<int>, MinSequenceContainer<double>>();
  test<std::vector<int, min_allocator<int>>, std::vector<double, min_allocator<double>>>();
  {
    bool transparent_used = false;
    TransparentComparator c(transparent_used);
    std::flat_map<int, int, TransparentComparator> m(std::sorted_unique, {{1, 1}, {2, 2}, {3, 3}}, c);
    assert(!transparent_used);
    m[ConvertibleTransparent<int>{3}];
    assert(transparent_used);
  }
  {
    auto index_func = [](auto& m, auto key_arg, auto value_arg) {
      using FlatMap                             = std::decay_t<decltype(m)>;
      using Key                                 = typename FlatMap::key_type;
      const typename FlatMap::mapped_type value = value_arg;
      m[ConvertibleTransparent<Key>{key_arg}]   = value;
    };
    test_emplace_exception_guarantee(index_func);
  }
  return 0;
}
