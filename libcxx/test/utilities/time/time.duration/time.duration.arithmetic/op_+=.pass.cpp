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

// duration& operator+=(const duration& d);

#include <chrono>
#include <cassert>

int main()
{
    std::chrono::seconds s(3);
    s += std::chrono::seconds(2);
    assert(s.count() == 5);
    s += std::chrono::minutes(2);
    assert(s.count() == 125);
}
