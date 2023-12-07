//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iostream>

// wostream wcerr;

// UNSUPPORTED: no-wide-characters
// REQUIRES: target={{.+}}-windows-{{.+}}

// UNSUPPORTED: executor-has-no-bash
// FILE_DEPENDENCIES: check-stderr.sh, test.dat
// RUN: %{build}
// RUN: %{exec} bash check-stderr.sh "%t.exe" "test.dat"

// Check that wcerr works, preserving the unicode characters, after switching
// stderr to wide mode.

#include <iostream>
#include <io.h>
#include <fcntl.h>

int main(int, char**) {
    _setmode(_fileno(stderr), _O_WTEXT);
    std::wcerr << L"1234\u20ac\u00e5\u00e4\u00f6";
    return 0;
}
