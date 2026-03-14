//===-- Unittests for mblen -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "src/wchar/mblen.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/Test.h"

using LlvmLibcMBLenTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcMBLenTest, OneByte) {
  const char *ch = "A";
  int n = LIBC_NAMESPACE::mblen(ch, 1);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_EQ(n, 1);

  // Should fail since we have not read enough
  n = LIBC_NAMESPACE::mblen(ch, 0);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_EQ(n, -1);
}

TEST_F(LlvmLibcMBLenTest, TwoByte) {
  const char ch[2] = {static_cast<char>(0xC2),
                      static_cast<char>(0x8E)}; //  car symbol
  int n = LIBC_NAMESPACE::mblen(ch, 4);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_EQ(n, 2);

  // Should fail since we have not read enough
  n = LIBC_NAMESPACE::mblen(ch, 1);
  ASSERT_EQ(n, -1);
  ASSERT_ERRNO_SUCCESS();
  // Should fail after trying to read next byte too
  n = LIBC_NAMESPACE::mblen(ch + 1, 1);
  ASSERT_EQ(n, -1);
  // This one should be an invalid starting byte so should set errno
  ASSERT_ERRNO_EQ(EILSEQ);
}

TEST_F(LlvmLibcMBLenTest, ThreeByte) {
  const char ch[3] = {static_cast<char>(0xE2), static_cast<char>(0x88),
                      static_cast<char>(0x91)}; // ∑ sigma symbol
  int n = LIBC_NAMESPACE::mblen(ch, 3);
  ASSERT_EQ(n, 3);
  ASSERT_ERRNO_SUCCESS();

  // Should fail since we have not read enough
  n = LIBC_NAMESPACE::mblen(ch, 2);
  ASSERT_EQ(n, -1);
  ASSERT_ERRNO_SUCCESS();
}

TEST_F(LlvmLibcMBLenTest, FourByte) {
  const char ch[4] = {static_cast<char>(0xF0), static_cast<char>(0x9F),
                      static_cast<char>(0xA4),
                      static_cast<char>(0xA1)}; // 🤡 clown emoji
  int n = LIBC_NAMESPACE::mblen(ch, 4);
  ASSERT_EQ(n, 4);
  ASSERT_ERRNO_SUCCESS();

  // Should fail since we have not read enough
  n = LIBC_NAMESPACE::mblen(ch, 2);
  ASSERT_EQ(n, -1);
  ASSERT_ERRNO_SUCCESS();
}

TEST_F(LlvmLibcMBLenTest, InvalidByte) {
  const char ch[1] = {static_cast<char>(0x80)};
  int n = LIBC_NAMESPACE::mblen(ch, 1);
  ASSERT_EQ(n, -1);
  ASSERT_ERRNO_EQ(EILSEQ);
}

TEST_F(LlvmLibcMBLenTest, InvalidMultiByte) {
  const char ch[4] = {static_cast<char>(0x80), static_cast<char>(0x00),
                      static_cast<char>(0x80),
                      static_cast<char>(0x00)}; // invalid sequence of bytes
  // Trying to push all 4 should error
  int n = LIBC_NAMESPACE::mblen(ch, 4);
  ASSERT_EQ(n, -1);
  ASSERT_ERRNO_EQ(EILSEQ);

  // Trying to push the second and third should correspond to null wc
  n = LIBC_NAMESPACE::mblen(ch + 1, 2);
  ASSERT_EQ(n, 0);
  ASSERT_ERRNO_SUCCESS();
}

TEST_F(LlvmLibcMBLenTest, NullString) {
  // reading on nullptr should return 0
  int n = LIBC_NAMESPACE::mblen(nullptr, 2);
  ASSERT_EQ(n, 0);
  ASSERT_ERRNO_SUCCESS();
  // reading a null terminator should return 0
  const char *ch = "\0";
  n = LIBC_NAMESPACE::mblen(ch, 1);
  ASSERT_EQ(n, 0);
}
