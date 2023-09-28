//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// Make sure that std::cbegin(x) effectively calls std::as_const(x).begin(), not x.cbegin().
//
// Also make sure that we don't get hijacked by ADL, see https://llvm.org/PR28927.

#include <cassert>
#include <iterator>

#include "test_macros.h"

struct ArrayHijacker {
  friend constexpr int begin(ArrayHijacker (&)[3]) { return 42; }
  friend constexpr int end(ArrayHijacker (&)[3]) { return 42; }
  friend constexpr int begin(const ArrayHijacker (&)[3]) { return 42; }
  friend constexpr int end(const ArrayHijacker (&)[3]) { return 42; }
};

struct ContainerHijacker {
  int* a_;
  constexpr int* begin() const { return a_; }
  constexpr int* end() const { return a_ + 3; }
  constexpr int* rbegin() const { return a_; }
  constexpr int* rend() const { return a_ + 3; }
  friend constexpr int begin(ContainerHijacker&) { return 42; }
  friend constexpr int end(ContainerHijacker&) { return 42; }
  friend constexpr int begin(const ContainerHijacker&) { return 42; }
  friend constexpr int end(const ContainerHijacker&) { return 42; }
  friend constexpr int cbegin(ContainerHijacker&) { return 42; }
  friend constexpr int cend(ContainerHijacker&) { return 42; }
  friend constexpr int cbegin(const ContainerHijacker&) { return 42; }
  friend constexpr int cend(const ContainerHijacker&) { return 42; }
  friend constexpr int rbegin(ContainerHijacker&) { return 42; }
  friend constexpr int rend(ContainerHijacker&) { return 42; }
  friend constexpr int rbegin(const ContainerHijacker&) { return 42; }
  friend constexpr int rend(const ContainerHijacker&) { return 42; }
  friend constexpr int crbegin(ContainerHijacker&) { return 42; }
  friend constexpr int crend(ContainerHijacker&) { return 42; }
  friend constexpr int crbegin(const ContainerHijacker&) { return 42; }
  friend constexpr int crend(const ContainerHijacker&) { return 42; }
};

TEST_CONSTEXPR_CXX17 bool test() {
  {
    ArrayHijacker a[3] = {};
    assert(begin(a) == 42);
    assert(end(a) == 42);
    assert(std::begin(a) == a);
    assert(std::end(a) == a + 3);
#if TEST_STD_VER > 11
    assert(std::cbegin(a) == a);
    assert(std::cend(a) == a + 3);
    assert(std::rbegin(a).base() == a + 3);
    assert(std::rend(a).base() == a);
    assert(std::crbegin(a).base() == a + 3);
    assert(std::crend(a).base() == a);
#endif
  }
  {
    int a[3] = {};
    ContainerHijacker c{a};
    assert(begin(c) == 42);
    assert(end(c) == 42);
    assert(std::begin(c) == a);
    assert(std::end(c) == a + 3);
#if TEST_STD_VER > 11
    assert(std::cbegin(c) == a);
    assert(std::cend(c) == a + 3);
    assert(std::rbegin(c) == a);
    assert(std::rend(c) == a + 3);
    assert(std::crbegin(c) == a);
    assert(std::crend(c) == a + 3);
#endif
  }
  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 17
  static_assert(test());
#endif

  return 0;
}
