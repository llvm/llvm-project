//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iostream>

// wostream wcout;

// UNSUPPORTED: no-wide-characters

// RUN: %{build}
// RUN: %{exec} %t.exe > %t.actual
// RUN: echo -n 1234 > %t.expected
// RUN: diff %t.expected %t.actual

#include <iostream>

int main(int, char**) {
    std::wcout << L"1234";
    return 0;
}
