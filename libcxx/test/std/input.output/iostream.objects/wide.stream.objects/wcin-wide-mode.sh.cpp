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
// REQUIRES: target={{.+}}-windows-{{.+}}

// FILE_DEPENDENCIES: test.dat
// RUN: %{build}
// RUN: cat test.dat | %{exec} %t.exe

// Check that wcin works, preserving the unicode characters, after switching
// stdin to wide mode.

#include <iostream>
#include <cassert>
#include <io.h>
#include <fcntl.h>

int main(int, char**) {
    _setmode(_fileno(stdin), _O_WTEXT);
    std::wstring str;
    std::wcin >> str;
    assert(str == L"1234\u20ac\u00e5\u00e4\u00f6");
    return 0;
}
