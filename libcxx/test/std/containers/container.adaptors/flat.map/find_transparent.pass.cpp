//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// template<class K> iterator       find(const K& x);
// template<class K> const_iterator find(const K& x) const;

#include <cassert>
#include <flat_map>
#include <string>
#include <utility>

#include "helpers.h"
#include "test_macros.h"

int main(int, char**) {
  {
    using M = std::flat_map<std::string, int, StartsWith::Less>;
    M m     = {{"alpha", 1}, {"beta", 2}, {"epsilon", 3}, {"eta", 4}, {"gamma", 5}};
    ASSERT_SAME_TYPE(decltype(m.find(StartsWith('b'))), M::iterator);
    ASSERT_SAME_TYPE(decltype(std::as_const(m).find(StartsWith('b'))), M::const_iterator);
    assert(m.find("beta") == m.begin() + 1);
    assert(m.find("delta") == m.end());
    assert(m.find("zeta") == m.end());
    assert(m.find(StartsWith('b')) == m.begin() + 1);
    assert(m.find(StartsWith('d')) == m.end());
    auto it = m.find(StartsWith('e'));
    assert(m.begin() + 2 <= it && it <= m.begin() + 3); // either is acceptable
    LIBCPP_ASSERT(it == m.begin() + 2);                 // return the earliest match
    assert(m.find(StartsWith('z')) == m.end());
  }
  return 0;
}
