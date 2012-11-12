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
    {
    const std::chrono::minutes m(3);
    std::chrono::minutes m2 = +m;
    assert(m.count() == m2.count());
    }
#ifndef _LIBCPP_HAS_NO_CONSTEXPR
    {
    constexpr std::chrono::minutes m(3);
    constexpr std::chrono::minutes m2 = +m;
    static_assert(m.count() == m2.count(), "");
    }
#endif
}
