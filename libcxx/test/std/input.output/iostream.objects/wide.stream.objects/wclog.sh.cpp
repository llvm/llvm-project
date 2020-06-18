//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iostream>

// istream wclog;

// RUN: %{build}
// RUN: %{exec} %t.exe 2> %t.err
// RUN: grep -e 'Hello World!' %t.err

#include <iostream>

#include "test_macros.h"

int main(int, char**)
{
    std::wclog << L"Hello World!\n";

    return 0;
}
