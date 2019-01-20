//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// void_t

#include <type_traits>

#include "test_macros.h"

#if TEST_STD_VER <= 14
# ifdef __cpp_lib_void_t
#   error Feature test macro should not be defined!
# endif
#else
# ifndef __cpp_lib_void_t
#   error Feature test macro is not defined
# endif
# if __cpp_lib_void_t != 201411
#   error Feature test macro has the wrong value
# endif
#endif

int main()
{
#if defined(__cpp_lib_void_t)
  static_assert(std::is_same_v<std::void_t<int>, void>, "");
#endif
}
