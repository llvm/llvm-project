//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <ios>

// typedef OFF_T streamoff;

#include <ios>
#include <type_traits>

int main(int, char**)
{
    static_assert(std::is_integral<std::streamoff>::value, "");
    static_assert(std::is_signed<std::streamoff>::value, "");

  return 0;
}
