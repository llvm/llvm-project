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
  wchar_t str[5];
  for (int i = 0; i < 5; i++)
    str[i] = 'A';

  wchar_t *output = LIBC_NAMESPACE::wmemset(str + 1, 'B', 3);

  EXPECT_EQ(output, str + 1);

  // EXPECT_TRUE being used since there isn't currently support for printing
  // wide chars in the future, it would be preferred to switch these to
  // EXPECT_EQ
  EXPECT_TRUE(str[0] == (wchar_t)'A');
  EXPECT_TRUE(str[1] == (wchar_t)'B');
  EXPECT_TRUE(str[2] == (wchar_t)'B');
  EXPECT_TRUE(str[3] == (wchar_t)'B');
  EXPECT_TRUE(str[4] == (wchar_t)'A');
}

TEST(LlvmLibcWMemsetTest, LargeStringBoundCheck) {
  constexpr int str_size = 1000;
  wchar_t str[str_size];
  for (int i = 0; i < str_size; i++)
    str[i] = 'A';

  wchar_t *output = LIBC_NAMESPACE::wmemset(str + 1, 'B', str_size - 2);

  EXPECT_EQ(output, str + 1);

  EXPECT_TRUE(str[0] == (wchar_t)'A');
  for (int i = 1; i < str_size - 1; i++)
    EXPECT_TRUE(str[i] == (wchar_t)'B');

  EXPECT_TRUE(str[str_size - 1] == (wchar_t)'A');
}

TEST(LlvmLibcWMemsetTest, WCharSizeSmallString) {
  // ensure we can handle full range of widechars
  wchar_t str[5];
  const wchar_t target = WCHAR_MAX;

  for (int i = 0; i < 5; i++)
    str[i] = 'A';

  wchar_t *output = LIBC_NAMESPACE::wmemset(str + 1, target, 3);

  EXPECT_EQ(output, str + 1);

  EXPECT_TRUE(str[0] == (wchar_t)'A');
  EXPECT_TRUE(str[1] == target);
  EXPECT_TRUE(str[2] == target);
  EXPECT_TRUE(str[3] == target);
  EXPECT_TRUE(str[4] == (wchar_t)'A');
}

TEST(LlvmLibcWMemsetTest, WCharSizeLargeString) {
  // ensure we can handle full range of widechars
  constexpr int str_size = 1000;
  wchar_t str[str_size];

  const wchar_t target = WCHAR_MAX;

  for (int i = 0; i < str_size; i++)
    str[i] = 'A';

  wchar_t *output = LIBC_NAMESPACE::wmemset(str + 1, target, str_size - 2);

  EXPECT_EQ(output, str + 1);

  EXPECT_TRUE(str[0] == (wchar_t)'A');
  for (int i = 1; i < str_size - 1; i++)
    EXPECT_TRUE(str[i] == target);

  EXPECT_TRUE(str[str_size - 1] == (wchar_t)'A');
}
