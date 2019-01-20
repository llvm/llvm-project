//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: libcpp-has-no-stdout

// <iostream>

// istream cout;

#include <iostream>

int main()
{
#if 0
    std::cout << "Hello World!\n";
    int i;
    std::cout << "Enter a number: ";
    std::cin >> i;
    std::cout << "The number is : " << i << '\n';
#else  // 0
    (void)std::cout;
#endif
}
