//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iostream>

// istream wcerr;

// RUN: %{build}
// RUN: %{exec} %t.exe 2> %t.err
// RUN: grep -e 'Hello World!' %t.err

#include <iostream>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    std::wcerr << L"Hello World!\n";

#ifdef _LIBCPP_HAS_NO_STDOUT
    assert(std::wcerr.tie() == NULL);
#else
    assert(std::wcerr.tie() == &std::wcout);
#endif
    assert(std::wcerr.flags() & std::ios_base::unitbuf);

    return 0;
}
