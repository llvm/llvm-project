//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iostream>

// wostream wcerr;

// XFAIL: no-wide-characters

// RUN: %{build}
// RUN: %{exec} %t.exe 2> %t.actual
// RUN: echo -n 1234 > %t.expected
// RUN: diff %t.expected %t.actual

#include <iostream>
#include <cassert>

#include "test_macros.h"

int main(int, char**) {
    std::wcerr << L"1234";
    assert(std::wcerr.flags() & std::ios_base::unitbuf);
    assert(std::wcerr.tie() == &std::wcout);
    return 0;
}
