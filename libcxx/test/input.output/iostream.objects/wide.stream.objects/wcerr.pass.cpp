//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <iostream>

// istream wcerr;

#include <iostream>
#include <cassert>

int main()
{
#if 0
    std::wcerr << L"Hello World!\n";
#else
    assert(std::wcerr.tie() == &std::wcout);
    assert(std::wcerr.flags() & std::ios_base::unitbuf);
#endif  // 0
}
