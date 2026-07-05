//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: LIBCXX-PICOLIBC-FIXME

// <iostream>

// ostream cerr;

// RUN: %{build}
// RUN: %{exec} %t.exe 2> %t.actual
// RUN: echo -n 1234 > %t.expected
// RUN: diff %t.expected %t.actual

#include <iostream>
#include <cassert>

#include "test_macros.h"

int main(int, char**) {
    std::cerr << "1234";
    assert(std::cerr.flags() & std::ios_base::unitbuf);
    assert(std::cerr.tie() == &std::cout);
    return 0;
}
