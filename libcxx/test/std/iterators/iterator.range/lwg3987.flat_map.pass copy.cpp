//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// UNSUPPORTED: clang-modules-build

// <flat_map>

// In addition to being available via inclusion of the <iterator> header,
// the function templates in [iterator.range] are available when any of the following
// headers are included: <flat_map>

#include <flat_map>

#include <cassert>

#include "test_macros.h"

constexpr bool test() {
  {
    std::flat_map<int, int> m{{1, 1}, {2, 2}};
    const auto& cm = m;
    assert(std::begin(m) == m.begin());
    assert(std::begin(cm) == cm.begin());
    assert(std::end(m) == m.end());
    assert(std::end(cm) == cm.end());
    assert(std::cbegin(m) == cm.begin());
    assert(std::cbegin(cm) == cm.begin());
    assert(std::cend(m) == cm.end());
    assert(std::cend(cm) == cm.end());
    assert(std::rbegin(m) == m.rbegin());
    assert(std::rbegin(cm) == cm.rbegin());
    assert(std::rend(m) == cm.rend());
    assert(std::rend(cm) == cm.rend());
    assert(std::crbegin(m) == cm.rbegin());
    assert(std::crbegin(cm) == cm.rbegin());
    assert(std::crend(m) == cm.rend());
    assert(std::crend(cm) == cm.rend());
    assert(std::size(cm) == cm.size());
    assert(std::ssize(cm) == decltype(std::ssize(m))(m.size()));
    assert(std::empty(cm) == cm.empty());
  }
  {
    int a[] = {1, 2, 3};
    assert(std::begin(a) == &a[0]);
    assert(std::end(a) == &a[3]);
    assert(std::rbegin(a) == std::reverse_iterator<int*>(std::end(a)));
    assert(std::rend(a) == std::reverse_iterator<int*>(std::begin(a)));
    assert(std::size(a) == 3);
    assert(std::ssize(a) == 3);
    assert(!std::empty(a));
    assert(std::data(a) == &a[0]);
  }
  {
    auto a = {1, 2, 3};
    assert(std::rbegin(a) == std::reverse_iterator<const int*>(std::end(a)));
    assert(std::rend(a) == std::reverse_iterator<const int*>(std::begin(a)));
    assert(!std::empty(a));
    assert(std::data(a) == &*std::begin(a));
  }

  return true;
}

int main(int, char**) {
  test();

  return 0;
}