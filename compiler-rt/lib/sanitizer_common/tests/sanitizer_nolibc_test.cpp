//===-- sanitizer_nolibc_test.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer/AddressSanitizer runtime.
// Tests for libc independence of sanitizer_common.
//
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_platform.h"

#include "gtest/gtest.h"

#include <stdlib.h>

extern const char *argv0;

// When SANITIZER_AMDHSA is enabled, CMake defines this macro and does not build
// Sanitizer-*-Test-Nolibc (see tests/CMakeLists.txt).
#if SANITIZER_LINUX && defined(__x86_64__) && \
    !defined(COMPILER_RT_SKIP_NOLIBC_SUBPROCESS_TEST)
TEST(SanitizerCommon, NolibcMain) {
  std::string NolibcTestPath = argv0;
  NolibcTestPath += "-Nolibc";
  int status = system(NolibcTestPath.c_str());
  EXPECT_EQ(true, WIFEXITED(status));
  EXPECT_EQ(0, WEXITSTATUS(status));
}
#endif
