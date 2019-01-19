//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test ratio:  The template argument D shall not be zero

#include <ratio>
#include <cstdint>

int main()
{
    const std::intmax_t t1 = std::ratio<1, 0>::num;
}
