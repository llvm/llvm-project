//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: libcpp-has-no-stdin

// <iostream>

// istream cin;

// RUN: %{build}
// RUN: %{exec} echo "123" \| %t.exe > %t.out
// RUN: grep -e 'The number is 123!' %t.out

#include <iostream>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    int i;
    std::cin >> i;
    std::cout << "The number is " << i << "!";

#ifdef _LIBCPP_HAS_NO_STDOUT
    assert(std::cin.tie() == NULL);
#else
    assert(std::cin.tie() == &std::cout);
#endif

  return 0;
}
