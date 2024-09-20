//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// template<class K> size_type count(const K& x) const;

#include <cassert>
#include <flat_map>
#include <string>
#include <utility>

#include "../helpers.h"
#include "test_macros.h"

// Constraints: The qualified-id Compare::is_transparent is valid and denotes a type.
template <class M>
concept CanCount        = requires(M m, Transparent<int> k) { m.count(k); };
using TransparentMap    = std::flat_map<int, double, TransparentComparator>;
using NonTransparentMap = std::flat_map<int, double, NonTransparentComparator>;
static_assert(CanCount<TransparentMap>);
static_assert(CanCount<const TransparentMap>);
static_assert(!CanCount<NonTransparentMap>);
static_assert(!CanCount<const NonTransparentMap>);

int main(int, char**) {
  {
    using M = std::flat_map<std::string, int, TransparentComparator>;
    M m     = {{"alpha", 1}, {"beta", 2}, {"epsilon", 3}, {"eta", 4}, {"gamma", 5}};
    ASSERT_SAME_TYPE(decltype(m.count(Transparent<std::string>{"abc"})), M::size_type);
    ASSERT_SAME_TYPE(decltype(std::as_const(m).count(Transparent<std::string>{"b"})), M::size_type);
    assert(m.count(Transparent<std::string>{"alpha"}) == 1);
    assert(m.count(Transparent<std::string>{"beta"}) == 1);
    assert(m.count(Transparent<std::string>{"epsilon"}) == 1);
    assert(m.count(Transparent<std::string>{"eta"}) == 1);
    assert(m.count(Transparent<std::string>{"gamma"}) == 1);
    assert(m.count(Transparent<std::string>{"al"}) == 0);
    assert(m.count(Transparent<std::string>{""}) == 0);
    assert(m.count(Transparent<std::string>{"g"}) == 0);
  }
#if 0
// do we really want to support this weird comparator that gives different answer for Key and Kp?
  {
    using M = std::flat_map<std::string, int, StartsWith::Less>;
    M m     = {{"alpha", 1}, {"beta", 2}, {"epsilon", 3}, {"eta", 4}, {"gamma", 5}};
    ASSERT_SAME_TYPE(decltype(m.count(StartsWith('b'))), M::size_type);
    ASSERT_SAME_TYPE(decltype(std::as_const(m).count(StartsWith('b'))), M::size_type);
    assert(m.count("beta") == 1);
    assert(m.count("delta") == 0);
    assert(m.count("zeta") == 0);
    assert(m.count(StartsWith('b')) == 1);
    assert(m.count(StartsWith('d')) == 0);
    assert(m.count(StartsWith('e')) == 2);
    assert(m.count(StartsWith('z')) == 0);
  }
#endif
  return 0;
}
