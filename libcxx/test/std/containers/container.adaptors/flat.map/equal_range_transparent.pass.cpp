//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// template<class K> pair<iterator,iterator>             equal_range(const K& x);
// template<class K> pair<const_iterator,const_iterator> equal_range(const K& x) const;

#include <cassert>
#include <flat_map>
#include <string>
#include <utility>

#include "test_macros.h"

struct StartsWith {
  explicit StartsWith(char ch) : lower_(1, ch), upper_(1, ch+1) {}
  StartsWith(const StartsWith&) = delete;
  void operator=(const StartsWith&) = delete;
  struct Less {
    using is_transparent = void;
    bool operator()(const std::string& a, const std::string& b) const { return a < b; }
    bool operator()(const StartsWith& a, const std::string& b) const { return a.upper_ <= b; }
    bool operator()(const std::string& a, const StartsWith& b) const { return a < b.lower_; }
  };
private:
  std::string lower_;
  std::string upper_;
};

int main(int, char**)
{
  {
    using M = std::flat_map<std::string, int, StartsWith::Less>;
    using R = std::pair<M::iterator, M::iterator>;
    using CR = std::pair<M::const_iterator, M::const_iterator>;
    M m = { {"alpha", 1}, {"beta", 2}, {"epsilon", 3}, {"eta", 4}, {"gamma", 5} };
    ASSERT_SAME_TYPE(decltype(m.equal_range(StartsWith('b'))), R);
    ASSERT_SAME_TYPE(decltype(std::as_const(m).equal_range(StartsWith('b'))), CR);
    auto begin = m.begin();
    assert(m.equal_range("beta") == std::pair(begin+1, begin+2));
    assert(m.equal_range("delta") == std::pair(begin+2, begin+2));
    assert(m.equal_range("zeta") == std::pair(begin+5, begin+5));
    assert(m.equal_range(StartsWith('b')) == std::pair(begin+1, begin+2));
    assert(m.equal_range(StartsWith('d')) == std::pair(begin+2, begin+2));
    assert(m.equal_range(StartsWith('e')) == std::pair(begin+2, begin+4));
    assert(m.equal_range(StartsWith('z')) == std::pair(begin+5, begin+5));
  }
  return 0;
}
