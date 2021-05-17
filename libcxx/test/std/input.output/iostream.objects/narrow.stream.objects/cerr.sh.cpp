//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iostream>

// istream cerr;

// XFAIL: LIBCXX-WINDOWS-FIXME

// FILE_DEPENDENCIES: ../check-stderr.sh
// RUN: %{build}
// RUN: %{exec} bash check-stderr.sh "%t.exe" "1234"

#include <iostream>
#include <cassert>

#include "test_macros.h"

int main(int, char**) {
    std::cerr << "1234";
    assert(std::cerr.flags() & std::ios_base::unitbuf);

#ifdef _LIBCPP_HAS_NO_STDOUT
    assert(std::cerr.tie() == NULL);
#else
    assert(std::cerr.tie() == &std::cout);
#endif
    return 0;
}
