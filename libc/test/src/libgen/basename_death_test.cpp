//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Death tests for basename.
///
//===----------------------------------------------------------------------===//

#include "src/libgen/basename.h"
#include "test/UnitTest/Test.h"

#ifdef ENABLE_SUBPROCESS_TESTS
TEST(LlvmLibcBasenameTest, ModifyReturnValue) {
  char *r = LIBC_NAMESPACE::basename(nullptr);
  ASSERT_DEATH([r]() { r[0] = 'a'; }, WITH_SIGNAL(-1));
}
#endif
