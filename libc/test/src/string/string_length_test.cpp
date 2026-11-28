//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// \file
// This file contains unit tests for string_length
//
//===----------------------------------------------------------------------===//

#include "hdr/stdint_proxy.h"
#include "src/string/string_length.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcStringLengthTest, HasZeroes64BitWordBlockTrue) {
  bool result =
      LIBC_NAMESPACE::internal::word::has_zeroes<uint64_t>(0x0011223344556677);
  ASSERT_TRUE(result);
}

TEST(LlvmLibcStringLengthTest, HasZeroes64BitWordFalse) {
  bool result =
      LIBC_NAMESPACE::internal::word::has_zeroes<uint64_t>(0x1122334455667788);
  ASSERT_FALSE(result);
}

TEST(LlvmLibcStringLengthTest, HasZeroes32BitWordBlockTrue) {
  bool result =
      LIBC_NAMESPACE::internal::word::has_zeroes<uint32_t>(0x00112233);
  ASSERT_TRUE(result);
}

TEST(LlvmLibcStringLengthTest, HasZeroes32BitWordBlockFalse) {
  bool result =
      LIBC_NAMESPACE::internal::word::has_zeroes<uint32_t>(0x11223344);
  ASSERT_FALSE(result);
}
