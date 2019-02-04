//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++98, c++03

// test quick_exit and at_quick_exit

#include <cstdlib>

void f() {}

int main(int, char**)
{
#ifdef _LIBCPP_HAS_QUICK_EXIT
    std::at_quick_exit(f);
    std::quick_exit(0);
#endif

  return 0;
}
