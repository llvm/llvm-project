//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// template<class K> iterator       lower_bound(const K& x);
// template<class K> const_iterator lower_bound(const K& x) const;

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
    ASSERT_SAME_TYPE(decltype(m.lower_bound(StartsWith('b'))), M::iterator);
    ASSERT_SAME_TYPE(decltype(std::as_const(m).lower_bound(StartsWith('b'))), M::const_iterator);
    assert(m.lower_bound("beta") == m.begin() + 1);
    assert(m.lower_bound("delta") == m.begin() + 2);
    assert(m.lower_bound("zeta") == m.begin() + 5);
    assert(m.lower_bound(StartsWith('b')) == m.begin() + 1);
    assert(m.lower_bound(StartsWith('d')) == m.begin() + 2);
    assert(m.lower_bound(StartsWith('e')) == m.begin() + 2);
    assert(m.lower_bound(StartsWith('z')) == m.begin() + 5);
  }
  return 0;
}
