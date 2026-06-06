//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <map>

// class map

// allocator_type get_allocator() const // constexpr since C++26

#include <map>
#include <cassert>
#include <string>

#include "test_allocator.h"
#include "test_macros.h"

TEST_CONSTEXPR_CXX26 bool test() {
  typedef std::pair<const int, std::string> ValueType;
  {
    std::allocator<ValueType> alloc;
    const std::map<int, std::string> m(alloc);
    assert(m.get_allocator() == alloc);
  }
  {
    other_allocator<ValueType> alloc(1);
    const std::map<int, std::string, std::less<int>, other_allocator<ValueType> > m(alloc);
    assert(m.get_allocator() == alloc);
  }
  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 26
  static_assert(test());
#endif
  return 0;
}
