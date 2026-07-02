//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for unistd-macros.
///
//===----------------------------------------------------------------------===//

#include "include/llvm-libc-macros/unistd-macros.h"
#include "test/UnitTest/Test.h"

namespace LIBC_NAMESPACE {

TEST(LlvmLibcUnistdMacrosTest, VersionMacros) {
#ifdef __linux__
#ifndef _POSIX_THREADS
#error "_POSIX_THREADS is not defined on Linux"
#endif
  EXPECT_EQ(_POSIX_THREADS, 202405L);
#endif
}

} // namespace LIBC_NAMESPACE
