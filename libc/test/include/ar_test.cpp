//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for the struct ar_hdr type.
///
//===----------------------------------------------------------------------===//

#include "include/llvm-libc-types/struct_ar_hdr.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcArTest, HeaderIsSixtyBytes) {
  ASSERT_EQ(sizeof(struct ar_hdr), size_t(60));
  ASSERT_EQ(__builtin_offsetof(struct ar_hdr, ar_fmag), size_t(58));
}
