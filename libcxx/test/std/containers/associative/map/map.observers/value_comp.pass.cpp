//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <map>

// value_compare value_comp() const; // constexpr since C++26

#include <map>
#include <cassert>
#include <string>

#include "test_macros.h"

TEST_CONSTEXPR_CXX26 bool test() {
  typedef std::map<int, std::string> map_type;

  map_type m;
  std::pair<map_type::iterator, bool> p1 = m.insert(map_type::value_type(1, "abc"));
  std::pair<map_type::iterator, bool> p2 = m.insert(map_type::value_type(2, "abc"));

  const map_type& cm = m;

  assert(cm.value_comp()(*p1.first, *p2.first));
  assert(!cm.value_comp()(*p2.first, *p1.first));
  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 26
  static_assert(test());
#endif
  return 0;
}
