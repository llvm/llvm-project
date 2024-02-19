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
// REQUIRES: target={{.+}}-windows-{{.+}}

// FILE_DEPENDENCIES: test.dat
// RUN: %{build}
// RUN: %{exec} %t.exe > %t.actual
// RUN: diff test.dat %t.actual

// Check that wcout works, preserving the unicode characters, after switching
// stdout to wide mode.

#include <iostream>
#include <io.h>
#include <fcntl.h>

int main(int, char**) {
    _setmode(_fileno(stdout), _O_WTEXT);
    std::wcout << L"1234\u20ac\u00e5\u00e4\u00f6";
    return 0;
}
