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

  EXPECT_TRUE(str[0] == (wchar_t)'A');
  EXPECT_TRUE(str[1] == (wchar_t)'B');
  EXPECT_TRUE(str[2] == (wchar_t)'B');
  EXPECT_TRUE(str[3] == (wchar_t)'B');
  EXPECT_TRUE(str[4] == (wchar_t)'A');
}

TEST(LlvmLibcWMemsetTest, LargeStringBoundCheck) {
  const int str_size = 1000;
  wchar_t* str = new wchar_t[str_size];
  for (int i = 0; i < str_size; i++) {
    str[i] = 'A';
  }

  wchar_t* output = LIBC_NAMESPACE::wmemset(str + 1, 'B', str_size - 2);

  EXPECT_EQ(output, str + 1);

  EXPECT_TRUE(str[0] == (wchar_t)'A');
  for (int i = 1; i < str_size - 1; i++) {
    EXPECT_TRUE(str[i] == (wchar_t)'B');
  }
  EXPECT_TRUE(str[str_size - 1] == (wchar_t)'A');
}

TEST(LlvmLibcWMemsetTest, WChar_Size_Small) {
  // ensure we can handle 32 bit values 
  wchar_t* str = new wchar_t[5];
  const wchar_t magic = INT32_MAX;
  
  for (int i = 0; i < 5; i++) {
    str[i] = 'A';
  }

  wchar_t* output = LIBC_NAMESPACE::wmemset(str + 1, magic, 3);

  EXPECT_EQ(output, str + 1);

  EXPECT_TRUE(str[0] == (wchar_t)'A');
  EXPECT_TRUE(str[1] == magic);
  EXPECT_TRUE(str[2] == magic);
  EXPECT_TRUE(str[3] == magic);
  EXPECT_TRUE(str[4] == (wchar_t)'A');
}

TEST(LlvmLibcWMemsetTest, WChar_Size_Large) {
  // ensure we can handle 32 bit values 
  const int str_size = 1000;
  const wchar_t magic = INT32_MAX;
  wchar_t* str = new wchar_t[str_size];
  for (int i = 0; i < str_size; i++) {
    str[i] = 'A';
  }

  wchar_t* output = LIBC_NAMESPACE::wmemset(str + 1, magic, str_size - 2);

  EXPECT_EQ(output, str + 1);

  EXPECT_TRUE(str[0] == (wchar_t)'A');
  for (int i = 1; i < str_size - 1; i++) {
    EXPECT_TRUE(str[i] == magic);
  }
  EXPECT_TRUE(str[str_size - 1] == (wchar_t)'A');
}


