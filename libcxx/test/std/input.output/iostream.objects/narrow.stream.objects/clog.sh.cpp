//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iostream>

// istream clog;

// XFAIL: LIBCXX-WINDOWS-FIXME

// FILE_DEPENDENCIES: ../check-stderr.sh
// RUN: %{build}
// RUN: %{exec} bash check-stderr.sh "%t.exe" "1234"

#include <iostream>

int main(int, char**) {
    std::clog << "1234";
    return 0;
}
