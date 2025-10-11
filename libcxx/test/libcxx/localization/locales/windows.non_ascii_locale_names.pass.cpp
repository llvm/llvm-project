//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// REQUIRES: windows

// The C RunTime library on Windows supports locale strings with
// characters outside the ASCII range. This poses challenges for
// code that temporarily set a custom thread locale.
//
// https://github.com/llvm/llvm-project/issues/160478

#include <iostream>
#include <iomanip>
#include <locale>
#include <clocale>
#include <cstdlib>
#include <cassert>

#include "test_macros.h"

int main(int, char**) {
    // Check that the C locale doesn't use the CP437 charset
    LIBCPP_ASSERT(std::setlocale(LC_ALL, "Norwegian Bokm\x86l_Norway") == nullptr);

    LIBCPP_ASSERT(std::setlocale(LC_ALL, ".437"));
    LIBCPP_ASSERT(std::setlocale(LC_ALL, "Norwegian Bokm\x86l_Norway.437"));

    std::cerr.imbue(std::locale::classic());
    std::cerr << std::setprecision(2) << 0.1 << std::endl;

    return EXIT_SUCCESS;
}
