//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <chrono>

// duration_values::max

#include <chrono>
#include <limits>
#include <cassert>

#include "../../rep.h"

int main()
{
    assert(std::chrono::duration_values<int>::max() ==
           std::numeric_limits<int>::max());
    assert(std::chrono::duration_values<double>::max() ==
           std::numeric_limits<double>::max());
    assert(std::chrono::duration_values<Rep>::max() ==
           std::numeric_limits<Rep>::max());
#ifndef _LIBCPP_HAS_NO_CONSTEXPR
    static_assert(std::chrono::duration_values<int>::max() ==
           std::numeric_limits<int>::max(), "");
    static_assert(std::chrono::duration_values<double>::max() ==
           std::numeric_limits<double>::max(), "");
    static_assert(std::chrono::duration_values<Rep>::max() ==
           std::numeric_limits<Rep>::max(), "");
#endif
}
