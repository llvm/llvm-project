//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for errx.
///
//===----------------------------------------------------------------------===//

#include "src/err/errx.h"
#include "test/UnitTest/Test.h"

namespace LIBC_NAMESPACE {

TEST(LlvmLibcErrxTest, ErrxExitCode) {
  EXPECT_EXITS([] { errx(2, "test errx"); }, 2);
}

TEST(LlvmLibcErrxTest, ErrxNullFormat) {
  EXPECT_EXITS([] { errx(2, nullptr); }, 2);
}

} // namespace LIBC_NAMESPACE
