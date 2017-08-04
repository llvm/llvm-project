//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <chrono>

// duration_values::min

#include <chrono>
#include <limits>
#include <cassert>

#include "test_macros.h"
#include "../../rep.h"

int main()
{
    assert(std::chrono::duration_values<int>::min() ==
           std::numeric_limits<int>::lowest());
    assert(std::chrono::duration_values<double>::min() ==
           std::numeric_limits<double>::lowest());
    assert(std::chrono::duration_values<Rep>::min() ==
           std::numeric_limits<Rep>::lowest());
#if TEST_STD_VER >= 11
    static_assert(std::chrono::duration_values<int>::min() ==
           std::numeric_limits<int>::lowest(), "");
    static_assert(std::chrono::duration_values<double>::min() ==
           std::numeric_limits<double>::lowest(), "");
    static_assert(std::chrono::duration_values<Rep>::min() ==
           std::numeric_limits<Rep>::lowest(), "");
#endif
}
