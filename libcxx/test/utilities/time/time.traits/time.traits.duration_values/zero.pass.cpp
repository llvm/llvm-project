//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <chrono>

// duration_values::zero

#include <chrono>
#include <cassert>

#include "../../rep.h"

int main()
{
    assert(std::chrono::duration_values<int>::zero() == 0);
    assert(std::chrono::duration_values<Rep>::zero() == 0);
#ifndef _LIBCPP_HAS_NO_CONSTEXPR
    static_assert(std::chrono::duration_values<int>::zero() == 0, "");
    static_assert(std::chrono::duration_values<Rep>::zero() == 0, "");
#endif
}
