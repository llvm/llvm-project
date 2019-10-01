//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iostream>

// istream wcerr;

#include <iostream>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
#if 0
    std::wcerr << L"Hello World!\n";
#else
#ifdef _LIBCPP_HAS_NO_STDOUT
    assert(std::wcerr.tie() == NULL);
#else
    assert(std::wcerr.tie() == &std::wcout);
#endif
    assert(std::wcerr.flags() & std::ios_base::unitbuf);
#endif  // 0

  return 0;
}
