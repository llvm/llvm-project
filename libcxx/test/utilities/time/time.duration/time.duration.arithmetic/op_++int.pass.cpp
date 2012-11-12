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

// duration operator++(int);

#include <chrono>
#include <cassert>

int main()
{
    std::chrono::hours h(3);
    std::chrono::hours h2 = h++;
    assert(h.count() == 4);
    assert(h2.count() == 3);
}
