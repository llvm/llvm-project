//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test ratio:  the absolute values of the template arguments N and D
//               shall be representable by type intmax_t.

#include <ratio>
#include <cstdint>

int main(int, char**)
{
    const std::intmax_t t1 = std::ratio<0x8000000000000000ULL, 1>::num;

  return 0;
}
