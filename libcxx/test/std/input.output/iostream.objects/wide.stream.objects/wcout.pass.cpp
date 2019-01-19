//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: libcpp-has-no-stdout

// <iostream>

// istream wcout;

#include <iostream>

int main()
{
#if 0
    std::wcout << L"Hello World!\n";
#else
    (void)std::wcout;
#endif
}
