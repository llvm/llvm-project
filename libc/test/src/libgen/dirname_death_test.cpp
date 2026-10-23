//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Death tests for dirname.
///
//===----------------------------------------------------------------------===//

#include "src/libgen/dirname.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcDirnameTest, ModifyReturnValue) {
  ASSERT_DEATH(
      []() {
        char *r = LIBC_NAMESPACE::dirname(nullptr);
        r[0] = 'a';
      },
      WITH_SIGNAL(-1));
}
