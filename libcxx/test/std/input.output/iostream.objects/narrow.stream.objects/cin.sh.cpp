//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// TODO: Investigate
// UNSUPPORTED: LIBCXX-AIX-FIXME

// <iostream>

// istream cin;

// RUN: %{build}
// RUN: echo -n 1234 | %{exec} %t.exe

#include <iostream>
#include <cassert>

int main(int, char**) {
    int i;
    std::cin >> i;
    assert(i == 1234);
    return 0;
}
