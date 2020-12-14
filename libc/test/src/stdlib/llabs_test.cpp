//===-- Unittests for llabs -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/llabs.h"
#include "utils/UnitTest/Test.h"

TEST(LlabsTest, Zero) { EXPECT_EQ(__llvm_libc::llabs(0ll), 0ll); }

TEST(LlabsTest, Positive) { EXPECT_EQ(__llvm_libc::llabs(1ll), 1ll); }

TEST(LlabsTest, Negative) { EXPECT_EQ(__llvm_libc::llabs(-1ll), 1ll); }
