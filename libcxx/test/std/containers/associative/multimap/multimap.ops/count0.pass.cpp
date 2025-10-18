//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// XFAIL: c++03, c++11

// <map>

// class multimap

// size_type count(const key_type& k) const; // constexpr since C++26

//
//   The member function templates find, count, lower_bound, upper_bound, and
// equal_range shall not participate in overload resolution unless the
// qualified-id Compare::is_transparent is valid and denotes a type

#include <map>
#include <cassert>

#include "test_macros.h"
#include "is_transparent.h"

TEST_CONSTEXPR_CXX26
bool test() {
  {
    typedef std::multimap<int, double, transparent_less> M;
    assert(M().count(C2Int{5}) == 0);
  }
  {
    typedef std::multimap<int, double, transparent_less_not_referenceable> M;
    assert(M().count(C2Int{5}) == 0);
  }
  {
    using M = std::multimap<int, double, transparent_less_nonempty>;
    assert(M().count(C2Int{5}) == 0);
  }

  return true;
}
int main(int, char**) {
  assert(test());

#if TEST_STD_VER >= 26
  static_assert(test());
#endif
  return 0;
}
