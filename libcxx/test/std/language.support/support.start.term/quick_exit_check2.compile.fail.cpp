//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++98, c++03

// test that referencing quick_exit when _LIBCPP_HAS_QUICK_EXIT is not defined
// results in a compile error.

#include <cstdlib>

void f() {}

int main(int, char**)
{
#ifndef _LIBCPP_HAS_QUICK_EXIT
    std::quick_exit(0);
#else
#error
#endif

  return 0;
}
