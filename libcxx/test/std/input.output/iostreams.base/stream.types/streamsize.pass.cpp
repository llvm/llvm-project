//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <ios>

// typedef SZ_T streamsize;

#include <ios>
#include <type_traits>

int main()
{
    static_assert(std::is_integral<std::streamsize>::value, "");
    static_assert(std::is_signed<std::streamsize>::value, "");
}
