//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// enable_if

#include <type_traits>

#include "test_macros.h"

int main()
{
    static_assert((std::is_same<std::enable_if<true>::type, void>::value), "");
    static_assert((std::is_same<std::enable_if<true, int>::type, int>::value), "");
#if TEST_STD_VER > 11
    static_assert((std::is_same<std::enable_if_t<true>, void>::value), "");
    static_assert((std::is_same<std::enable_if_t<true, int>, int>::value), "");
#endif
}
