//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <atomic>

// #define ATOMIC_BOOL_LOCK_FREE unspecified
// #define ATOMIC_CHAR_LOCK_FREE unspecified
// #define ATOMIC_CHAR8_T_LOCK_FREE unspecified // C++20
// #define ATOMIC_CHAR16_T_LOCK_FREE unspecified
// #define ATOMIC_CHAR32_T_LOCK_FREE unspecified
// #define ATOMIC_WCHAR_T_LOCK_FREE unspecified
// #define ATOMIC_SHORT_LOCK_FREE unspecified
// #define ATOMIC_INT_LOCK_FREE unspecified
// #define ATOMIC_LONG_LOCK_FREE unspecified
// #define ATOMIC_LLONG_LOCK_FREE unspecified
// #define ATOMIC_POINTER_LOCK_FREE unspecified

#include <atomic>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    assert(ATOMIC_BOOL_LOCK_FREE == 0 ||
           ATOMIC_BOOL_LOCK_FREE == 1 ||
           ATOMIC_BOOL_LOCK_FREE == 2);
    assert(ATOMIC_CHAR_LOCK_FREE == 0 ||
           ATOMIC_CHAR_LOCK_FREE == 1 ||
           ATOMIC_CHAR_LOCK_FREE == 2);
#if TEST_STD_VER > 17 && defined(__cpp_char8_t)
    assert(ATOMIC_CHAR8_T_LOCK_FREE == 0 ||
           ATOMIC_CHAR8_T_LOCK_FREE == 1 ||
           ATOMIC_CHAR8_T_LOCK_FREE == 2);
#endif
    assert(ATOMIC_CHAR16_T_LOCK_FREE == 0 ||
           ATOMIC_CHAR16_T_LOCK_FREE == 1 ||
           ATOMIC_CHAR16_T_LOCK_FREE == 2);
    assert(ATOMIC_CHAR32_T_LOCK_FREE == 0 ||
           ATOMIC_CHAR32_T_LOCK_FREE == 1 ||
           ATOMIC_CHAR32_T_LOCK_FREE == 2);
    assert(ATOMIC_WCHAR_T_LOCK_FREE == 0 ||
           ATOMIC_WCHAR_T_LOCK_FREE == 1 ||
           ATOMIC_WCHAR_T_LOCK_FREE == 2);
    assert(ATOMIC_SHORT_LOCK_FREE == 0 ||
           ATOMIC_SHORT_LOCK_FREE == 1 ||
           ATOMIC_SHORT_LOCK_FREE == 2);
    assert(ATOMIC_INT_LOCK_FREE == 0 ||
           ATOMIC_INT_LOCK_FREE == 1 ||
           ATOMIC_INT_LOCK_FREE == 2);
    assert(ATOMIC_LONG_LOCK_FREE == 0 ||
           ATOMIC_LONG_LOCK_FREE == 1 ||
           ATOMIC_LONG_LOCK_FREE == 2);
    assert(ATOMIC_LLONG_LOCK_FREE == 0 ||
           ATOMIC_LLONG_LOCK_FREE == 1 ||
           ATOMIC_LLONG_LOCK_FREE == 2);
    assert(ATOMIC_POINTER_LOCK_FREE == 0 ||
           ATOMIC_POINTER_LOCK_FREE == 1 ||
           ATOMIC_POINTER_LOCK_FREE == 2);

  return 0;
}
