//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: libcpp-has-no-stdout

// <iostream>

// istream cout;

// RUN: %{build}
// RUN: %{exec} %t.exe > %t.out
// RUN: grep -e 'Hello World!' %t.out

#include <iostream>

#include "test_macros.h"

int main(int, char**)
{
    std::cout << "Hello World!\n";

    return 0;
}
