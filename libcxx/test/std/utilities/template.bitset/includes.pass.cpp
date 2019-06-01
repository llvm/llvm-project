//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test that <bitset> includes <string> and <iosfwd>

#include <bitset>

#include "test_macros.h"

template <class> void test_typedef() {}

int main(int, char**)
{
  { // test for <string>
    std::string s; ((void)s);
  }
  { // test for <iosfwd>
    test_typedef<std::ios>();
    test_typedef<std::wios>();
    test_typedef<std::istream>();
    test_typedef<std::ostream>();
    test_typedef<std::iostream>();
  }

  return 0;
}
