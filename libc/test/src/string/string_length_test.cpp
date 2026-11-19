//===-- Unittests for string_length ----------------------------------- ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/string_length.h"
#include "hdr/stdint_proxy.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcStringLengthTest, Size64BitsHasZeroes) {
  bool result = LIBC_NAMESPACE::internal::word::has_zeroes<uint64_t>(0x0011223344556677);
  ASSERT_TRUE(result);
}
