//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iostream>

// wistream wcin;

// UNSUPPORTED: no-wide-characters

// RUN: %{build}
// RUN: echo -n 1234 > %t.input
// RUN: %{exec} %t.exe < %t.input

#include <iostream>
#include <cassert>

int main(int, char**) {
    int i;
    std::wcin >> i;
    assert(i == 1234);
    return 0;
}
