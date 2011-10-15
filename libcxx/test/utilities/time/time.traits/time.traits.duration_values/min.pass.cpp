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

#include "../../rep.h"

int main()
{
    assert(std::chrono::duration_values<int>::min() ==
           std::numeric_limits<int>::lowest());
    assert(std::chrono::duration_values<double>::min() ==
           std::numeric_limits<double>::lowest());
    assert(std::chrono::duration_values<Rep>::min() ==
           std::numeric_limits<Rep>::lowest());
}
