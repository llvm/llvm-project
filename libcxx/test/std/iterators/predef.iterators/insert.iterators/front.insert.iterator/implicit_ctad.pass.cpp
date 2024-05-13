//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14

// <iterator>

// front_insert_iterator

// Make sure that the implicitly-generated CTAD works.

#include <deque>
#include <iterator>
#include <string>

#include "test_macros.h"

int main(int, char**) {
  {
    std::string s;
    std::front_insert_iterator it(s);
    ASSERT_SAME_TYPE(decltype(it), std::front_insert_iterator<std::string>);
  }
  {
    std::deque<int> v;
    std::front_insert_iterator it(v);
    std::front_insert_iterator copy(it);
    ASSERT_SAME_TYPE(decltype(it), std::front_insert_iterator<std::deque<int>>);
    ASSERT_SAME_TYPE(decltype(copy), std::front_insert_iterator<std::deque<int>>);
  }

  return 0;
}
