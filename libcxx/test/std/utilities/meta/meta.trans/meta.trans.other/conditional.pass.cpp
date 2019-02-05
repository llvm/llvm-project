//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// conditional

#include <type_traits>

#include "test_macros.h"

int main(int, char**)
{
    static_assert((std::is_same<std::conditional<true, char, int>::type, char>::value), "");
    static_assert((std::is_same<std::conditional<false, char, int>::type, int>::value), "");
#if TEST_STD_VER > 11
    static_assert((std::is_same<std::conditional_t<true, char, int>, char>::value), "");
    static_assert((std::is_same<std::conditional_t<false, char, int>, int>::value), "");
#endif

  return 0;
}
