//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <ios>
//
// class ios_base
// {
// public:
//     typedef OFF_T streamoff;
// };

//  These members were removed for C++17

#include "test_macros.h"
#include <ios>
#include <type_traits>

int main(int, char**)
{
#if TEST_STD_VER <= 14
    static_assert((std::is_integral<std::ios_base::streamoff>::value), "");
    static_assert((std::is_signed<std::ios_base::streamoff>::value), "");
#endif

  return 0;
}
