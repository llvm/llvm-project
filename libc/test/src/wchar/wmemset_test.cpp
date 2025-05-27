//===-- Unittests for wmemset ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/types/size_t.h"
#include "hdr/types/wchar_t.h"
#include "src/wchar/wmemset.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcWMemsetTest, SmallStringBoundCheck) {
  wchar_t* str = new wchar_t[5];
  for (int i = 0; i < 5; i++) {
    str[i] = 'A';
  }

  wchar_t* output = LIBC_NAMESPACE::wmemset(str + 1, 'B', 3);

  EXPECT_EQ(output, str + 1);

  EXPECT_EQ(str[0], (wchar_t)'A');
  EXPECT_EQ(str[1], (wchar_t)'B');
  EXPECT_EQ(str[2], (wchar_t)'B');
  EXPECT_EQ(str[3], (wchar_t)'B');
  EXPECT_EQ(str[4], (wchar_t)'A');
}

