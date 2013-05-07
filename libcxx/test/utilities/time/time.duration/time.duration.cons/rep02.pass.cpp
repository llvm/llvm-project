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

// template <class Rep2>
//   explicit duration(const Rep2& r);

// construct double with int

#include <chrono>
#include <cassert>

int main()
{
    std::chrono::duration<double> d(5);
    assert(d.count() == 5);
#ifndef _LIBCPP_HAS_NO_CONSTEXPR
    constexpr std::chrono::duration<double> d2(5);
    static_assert(d2.count() == 5, "");
#endif
}
