//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// UNSUPPORTED: c++98, c++03, c++11, c++14
// UNSUPPORTED: libcpp-no-deduction-guides

// class back_insert_iterator.

#include <string>
#include <iterator>
#include <vector>
#include <cassert>
#include <cstddef>
#include <functional>

#include "test_macros.h"

int main(int, char**)
{
    {
      std::string s;
      std::back_insert_iterator bs(s);
      ASSERT_SAME_TYPE(decltype(bs), std::back_insert_iterator<std::string>);
    }
  {
    std::vector<int> v;
    std::back_insert_iterator bv(v);
    std::back_insert_iterator cp(bv);
    ASSERT_SAME_TYPE(decltype(bv), std::back_insert_iterator<std::vector<int>>);
    ASSERT_SAME_TYPE(decltype(cp), std::back_insert_iterator<std::vector<int>>);
  }

  return 0;
}
