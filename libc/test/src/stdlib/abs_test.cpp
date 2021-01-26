//===-- Unittests for abs -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/abs.h"
#include "utils/UnitTest/Test.h"

TEST(LlvmLibcAbsTest, Zero) { EXPECT_EQ(__llvm_libc::abs(0), 0); }

TEST(LlvmLibcAbsTest, Positive) { EXPECT_EQ(__llvm_libc::abs(1), 1); }

TEST(LlvmLibcAbsTest, Negative) { EXPECT_EQ(__llvm_libc::abs(-1), 1); }
