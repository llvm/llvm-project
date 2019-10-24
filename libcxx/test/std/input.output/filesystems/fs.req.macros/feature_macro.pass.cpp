//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <filesystem>

// #define __cpp_lib_filesystem 201703L

#include <filesystem>
#include "test_macros.h"

#if TEST_STD_VER >= 17
#ifndef __cpp_lib_filesystem
#error Filesystem feature test macro is not defined  (__cpp_lib_filesystem)
#elif __cpp_lib_filesystem != 201703L
#error Filesystem feature test macro has an incorrect value (__cpp_lib_filesystem)
#endif
#else // TEST_STD_VER < 17
#ifdef __cpp_lib_filesystem
#error Filesystem feature test macro should not be defined before C++17
#endif
#endif

int main(int, char**) {
  return 0;
}
