//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS

// <fstream>

#include <cassert>
#include <fstream>
#include <vector>

#include "test_macros.h"

void sputn_not_open() {
    std::vector<char> data(10, 'a');
    std::filebuf f;
    std::streamsize len = f.sputn(data.data(), data.size());
    assert(len == 0);
    assert(std::strncmp(data.data(), "aaaaaaaaaa", 10) == 0);
}

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
void sputn_not_open_wchar() {
    std::vector<wchar_t> data(10, L'a');
    std::wfilebuf f;
    std::streamsize len = f.sputn(data.data(), data.size());
    assert(len == 0);
    assert(std::wcsncmp(data.data(), L"aaaaaaaaaa", 10) == 0);
}
#endif

int main(int, char **) {
    sputn_not_open();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    sputn_not_open_wchar();
#endif
    return 0;
}