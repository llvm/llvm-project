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

// duration& operator%=(const duration& rhs)

#include <chrono>
#include <cassert>

int main()
{
    std::chrono::microseconds us(11);
    std::chrono::microseconds us2(3);
    us %= us2;
    assert(us.count() == 2);
    us %= std::chrono::milliseconds(3);
    assert(us.count() == 2);
}
