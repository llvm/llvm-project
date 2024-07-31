//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// template<class K> bool contains(const K& x) const;

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
    ASSERT_SAME_TYPE(decltype(m.contains(StartsWith('b'))), bool);
    ASSERT_SAME_TYPE(decltype(std::as_const(m).contains(StartsWith('b'))), bool);
    assert(m.contains("beta") == true);
    assert(m.contains("delta") == false);
    assert(m.contains("zeta") == false);
    assert(m.contains(StartsWith('b')) == true);
    assert(m.contains(StartsWith('d')) == false);
    assert(m.contains(StartsWith('e')) == true);
    assert(m.contains(StartsWith('z')) == false);
  }
  return 0;
}
