//===-- Unittests for mbrlen ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/types/wchar_t.h"
#include "src/__support/wchar/mbstate.h"
#include "src/string/memset.h"
#include "src/wchar/mbrlen.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/Test.h"

using LlvmLibcMBRLenTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcMBRLenTest, OneByte) {
  const char *ch = "A";
  mbstate_t mb;
  LIBC_NAMESPACE::memset(&mb, 0, sizeof(mbstate_t));
  size_t n = LIBC_NAMESPACE::mbrlen(ch, 1, &mb);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_EQ(n, static_cast<size_t>(1));

  // Should fail since we have not read enough
  n = LIBC_NAMESPACE::mbrlen(ch, 0, &mb);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_EQ(n, static_cast<size_t>(-2));
}

TEST_F(LlvmLibcMBRLenTest, TwoByte) {
  const char ch[2] = {static_cast<char>(0xC2),
                      static_cast<char>(0x8E)}; //  car symbol
  mbstate_t mb;
  LIBC_NAMESPACE::memset(&mb, 0, sizeof(mbstate_t));
  size_t n = LIBC_NAMESPACE::mbrlen(ch, 4, nullptr);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_EQ(static_cast<int>(n), 2);

  // Should fail since we have not read enough
  n = LIBC_NAMESPACE::mbrlen(ch, 1, &mb);
  ASSERT_EQ(static_cast<int>(n), -2);
  ASSERT_ERRNO_SUCCESS();
  // Should pass after trying to read next byte
  n = LIBC_NAMESPACE::mbrlen(ch + 1, 1, &mb);
  ASSERT_EQ(static_cast<int>(n), 1);
  ASSERT_ERRNO_SUCCESS();
}

TEST_F(LlvmLibcMBRLenTest, ThreeByte) {
  const char ch[3] = {static_cast<char>(0xE2), static_cast<char>(0x88),
                      static_cast<char>(0x91)}; // ∑ sigma symbol
  mbstate_t mb;
  LIBC_NAMESPACE::memset(&mb, 0, sizeof(mbstate_t));
  size_t n = LIBC_NAMESPACE::mbrlen(ch, 3, &mb);
  ASSERT_EQ(static_cast<int>(n), 3);
  ASSERT_ERRNO_SUCCESS();

  // Should fail since we have not read enough
  n = LIBC_NAMESPACE::mbrlen(ch, 2, &mb);
  ASSERT_EQ(static_cast<int>(n), -2);
  ASSERT_ERRNO_SUCCESS();
}

TEST_F(LlvmLibcMBRLenTest, FourByte) {
  const char ch[4] = {static_cast<char>(0xF0), static_cast<char>(0x9F),
                      static_cast<char>(0xA4),
                      static_cast<char>(0xA1)}; // 🤡 clown emoji
  mbstate_t mb;
  LIBC_NAMESPACE::memset(&mb, 0, sizeof(mbstate_t));
  size_t n = LIBC_NAMESPACE::mbrlen(ch, 4, &mb);
  ASSERT_EQ(static_cast<int>(n), 4);
  ASSERT_ERRNO_SUCCESS();

  // Should fail since we have not read enough
  n = LIBC_NAMESPACE::mbrlen(ch, 2, &mb);
  ASSERT_EQ(static_cast<int>(n), -2);
  ASSERT_ERRNO_SUCCESS();

  // Should fail since we have not read enough
  n = LIBC_NAMESPACE::mbrlen(ch + 2, 1, &mb);
  ASSERT_EQ(static_cast<int>(n), -2);
  ASSERT_ERRNO_SUCCESS();

  // Should pass after reading final byte
  n = LIBC_NAMESPACE::mbrlen(ch + 3, 5, &mb);
  ASSERT_EQ(static_cast<int>(n), 1);
  ASSERT_ERRNO_SUCCESS();
}

TEST_F(LlvmLibcMBRLenTest, InvalidByte) {
  const char ch[1] = {static_cast<char>(0x80)};
  size_t n = LIBC_NAMESPACE::mbrlen(ch, 1, nullptr);
  ASSERT_EQ(static_cast<int>(n), -1);
  ASSERT_ERRNO_EQ(EILSEQ);
}

TEST_F(LlvmLibcMBRLenTest, InvalidMultiByte) {
  const char ch[4] = {static_cast<char>(0x80), static_cast<char>(0x00),
                      static_cast<char>(0x80),
                      static_cast<char>(0x00)}; // invalid sequence of bytes
  mbstate_t mb;
  LIBC_NAMESPACE::memset(&mb, 0, sizeof(mbstate_t));
  // Trying to push all 4 should error
  size_t n = LIBC_NAMESPACE::mbrlen(ch, 4, &mb);
  ASSERT_EQ(static_cast<int>(n), -1);
  ASSERT_ERRNO_EQ(EILSEQ);

  // Trying to push the second and third should correspond to null wc
  n = LIBC_NAMESPACE::mbrlen(ch + 1, 2, &mb);
  ASSERT_EQ(static_cast<int>(n), 0);
  ASSERT_ERRNO_SUCCESS();
}

TEST_F(LlvmLibcMBRLenTest, NullString) {
  // reading on nullptr should return 0
  size_t n = LIBC_NAMESPACE::mbrlen(nullptr, 2, nullptr);
  ASSERT_EQ(static_cast<int>(n), 0);
  ASSERT_ERRNO_SUCCESS();
  // reading a null terminator should return 0
  const char *ch = "\0";
  n = LIBC_NAMESPACE::mbrlen(ch, 1, nullptr);
  ASSERT_EQ(static_cast<int>(n), 0);
}

TEST_F(LlvmLibcMBRLenTest, InvalidMBState) {
  const char ch[4] = {static_cast<char>(0xC2), static_cast<char>(0x8E),
                      static_cast<char>(0xC7), static_cast<char>(0x8C)};
  mbstate_t *mb;
  LIBC_NAMESPACE::internal::mbstate inv;
  inv.total_bytes = 6;
  mb = reinterpret_cast<mbstate_t *>(&inv);
  // invalid mbstate should error
  size_t n = LIBC_NAMESPACE::mbrlen(ch, 2, mb);
  ASSERT_EQ(static_cast<int>(n), -1);
  ASSERT_ERRNO_EQ(EINVAL);
}
