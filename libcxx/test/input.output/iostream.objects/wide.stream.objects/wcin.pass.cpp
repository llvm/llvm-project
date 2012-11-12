//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <iostream>

// istream wcin;

#include <iostream>
#include <cassert>

int main()
{
#if 0
    std::wcout << L"Hello World!\n";
    int i;
    std::wcout << L"Enter a number: ";
    std::wcin >> i;
    std::wcout << L"The number is : " << i << L'\n';
#else  // 0
    assert(std::wcin.tie() == &std::wcout);
#endif
}
