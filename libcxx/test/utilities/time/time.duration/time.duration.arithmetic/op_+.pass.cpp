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

// duration operator+() const;

#include <chrono>
#include <cassert>

int main()
{
    const std::chrono::minutes m(3);
    std::chrono::minutes m2 = +m;
    assert(m.count() == m2.count());
}
