//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// underlying_type

#include <type_traits>
#include <climits>

#include "test_macros.h"

enum E { V = INT_MIN };

#if !defined(_WIN32) || defined(__MINGW32__)
    #define TEST_UNSIGNED_UNDERLYING_TYPE 1
#else
    #define TEST_UNSIGNED_UNDERLYING_TYPE 0 // MSVC's ABI doesn't follow the Standard
#endif

#if TEST_UNSIGNED_UNDERLYING_TYPE
enum F { W = UINT_MAX };
#endif // TEST_UNSIGNED_UNDERLYING_TYPE

int main(int, char**)
{
    ASSERT_SAME_TYPE(int, std::underlying_type<E>::type);
#if TEST_UNSIGNED_UNDERLYING_TYPE
    ASSERT_SAME_TYPE(unsigned, std::underlying_type<F>::type);
#endif // TEST_UNSIGNED_UNDERLYING_TYPE

#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(int, std::underlying_type_t<E>);
#if TEST_UNSIGNED_UNDERLYING_TYPE
    ASSERT_SAME_TYPE(unsigned, std::underlying_type_t<F>);
#endif // TEST_UNSIGNED_UNDERLYING_TYPE
#endif // TEST_STD_VER > 11

#if TEST_STD_VER >= 11
    enum G : char { };

    ASSERT_SAME_TYPE(char,   std::underlying_type<G>::type);
#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(char, std::underlying_type_t<G>);
#endif // TEST_STD_VER > 11
#endif // TEST_STD_VER >= 11

  return 0;
}
