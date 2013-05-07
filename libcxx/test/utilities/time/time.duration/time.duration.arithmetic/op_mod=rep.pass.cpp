//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <chrono>

// duration

// duration& operator%=(const rep& rhs)

#include <chrono>
#include <cassert>

int main()
{
    std::chrono::microseconds us(11);
    us %= 3;
    assert(us.count() == 2);
}
