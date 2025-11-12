//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <map>

// key_compare key_comp() const; // constexpr since C++26

#include <map>
#include <cassert>
#include <string>

#include "test_macros.h"

TEST_CONSTEXPR_CXX26
bool test() {
  typedef std::multimap<int, std::string> map_type;

  map_type m;
  map_type::iterator i1 = m.insert(map_type::value_type(1, "abc"));
  map_type::iterator i2 = m.insert(map_type::value_type(2, "abc"));

  const map_type& cm = m;

  assert(cm.key_comp()(i1->first, i2->first));
  assert(!cm.key_comp()(i2->first, i1->first));

  return true;
}
int main(int, char**) {
  test();

#if TEST_STD_VER >= 26
  static_assert(test());
#endif
  return 0;
}
