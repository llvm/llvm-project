//===-- Unittests for mbtowc ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/types/wchar_t.h"
#include "src/wchar/mbtowc.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/Test.h"

using LlvmLibcMBToWCTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcMBToWCTest, OneByte) {
  const char *ch = "A";
  wchar_t dest[2];
  int n = LIBC_NAMESPACE::mbtowc(dest, ch, 1);
  ASSERT_EQ(static_cast<char>(*dest), 'A');
  ASSERT_EQ(n, 1);

  // Should fail since we have not read enough
  n = LIBC_NAMESPACE::mbtowc(dest, ch, 0);
  ASSERT_EQ(n, -1);
  ASSERT_ERRNO_EQ(EILSEQ);
}

TEST_F(LlvmLibcMBToWCTest, TwoByte) {
  const char ch[2] = {static_cast<char>(0xC2),
                      static_cast<char>(0x8E)}; //  car symbol
  wchar_t dest[2];
  int n = LIBC_NAMESPACE::mbtowc(dest, ch, 2);
  ASSERT_EQ(static_cast<int>(*dest), 142);
  ASSERT_EQ(n, 2);

  // Should fail since we have not read enough
  n = LIBC_NAMESPACE::mbtowc(dest, ch, 1);
  ASSERT_EQ(n, -1);
  // Should fail after trying to read next byte too
  n = LIBC_NAMESPACE::mbtowc(dest, ch + 1, 1);
  ASSERT_EQ(n, -1);
  ASSERT_ERRNO_EQ(EILSEQ);
}

TEST_F(LlvmLibcMBToWCTest, ThreeByte) {
  const char ch[3] = {static_cast<char>(0xE2), static_cast<char>(0x88),
                      static_cast<char>(0x91)}; // ∑ sigma symbol
  wchar_t dest[2];
  int n = LIBC_NAMESPACE::mbtowc(dest, ch, 3);
  ASSERT_EQ(static_cast<int>(*dest), 8721);
  ASSERT_EQ(n, 3);

  // Should fail since we have not read enough
  n = LIBC_NAMESPACE::mbtowc(dest, ch, 2);
  ASSERT_EQ(n, -1);
  ASSERT_ERRNO_EQ(EILSEQ);
}

TEST_F(LlvmLibcMBToWCTest, FourByte) {
  const char ch[4] = {static_cast<char>(0xF0), static_cast<char>(0x9F),
                      static_cast<char>(0xA4),
                      static_cast<char>(0xA1)}; // 🤡 clown emoji
  wchar_t dest[2];
  int n = LIBC_NAMESPACE::mbtowc(dest, ch, 4);
  ASSERT_EQ(static_cast<int>(*dest), 129313);
  ASSERT_EQ(n, 4);

  // Should fail since we have not read enough
  n = LIBC_NAMESPACE::mbtowc(dest, ch, 2);
  ASSERT_EQ(n, -1);
  ASSERT_ERRNO_EQ(EILSEQ);
}

TEST_F(LlvmLibcMBToWCTest, InvalidByte) {
  const char ch[1] = {static_cast<char>(0x80)};
  wchar_t dest[2];
  int n = LIBC_NAMESPACE::mbtowc(dest, ch, 1);
  ASSERT_EQ(n, -1);
  ASSERT_ERRNO_EQ(EILSEQ);
}

TEST_F(LlvmLibcMBToWCTest, InvalidMultiByte) {
  const char ch[4] = {static_cast<char>(0x80), static_cast<char>(0x00),
                      static_cast<char>(0x80),
                      static_cast<char>(0x00)}; // invalid sequence of bytes
  wchar_t dest[2];
  // Trying to push all 4 should error
  int n = LIBC_NAMESPACE::mbtowc(dest, ch, 4);
  ASSERT_EQ(n, -1);
  ASSERT_ERRNO_EQ(EILSEQ);

  // Trying to push the second and third should correspond to null wc
  n = LIBC_NAMESPACE::mbtowc(dest, ch + 1, 2);
  ASSERT_EQ(n, 0);
  ASSERT_TRUE(*dest == L'\0');
}

TEST_F(LlvmLibcMBToWCTest, InvalidLastByte) {
  // Last byte is invalid since it does not have correct starting sequence.
  // 0xC0 --> 11000000 starting sequence should be 10xxxxxx
  const char ch[4] = {static_cast<char>(0xF1), static_cast<char>(0x80),
                      static_cast<char>(0x80), static_cast<char>(0xC0)};
  wchar_t dest[2];
  // Trying to push all 4 should error
  int n = LIBC_NAMESPACE::mbtowc(dest, ch, 4);
  ASSERT_EQ(n, -1);
  ASSERT_ERRNO_EQ(EILSEQ);
}

TEST_F(LlvmLibcMBToWCTest, ValidTwoByteWithExtraRead) {
  const char ch[3] = {static_cast<char>(0xC2), static_cast<char>(0x8E),
                      static_cast<char>(0x80)};
  wchar_t dest[2];
  // Trying to push all 3 should return valid 2 byte
  int n = LIBC_NAMESPACE::mbtowc(dest, ch, 3);
  ASSERT_EQ(n, 2);
  ASSERT_EQ(static_cast<int>(*dest), 142);
}

TEST_F(LlvmLibcMBToWCTest, TwoValidTwoBytes) {
  const char ch[4] = {static_cast<char>(0xC2), static_cast<char>(0x8E),
                      static_cast<char>(0xC7), static_cast<char>(0x8C)};
  wchar_t dest[2];
  int n = LIBC_NAMESPACE::mbtowc(dest, ch, 2);
  ASSERT_EQ(n, 2);
  ASSERT_EQ(static_cast<int>(*dest), 142);
  n = LIBC_NAMESPACE::mbtowc(dest + 1, ch + 2, 2);
  ASSERT_EQ(n, 2);
  ASSERT_EQ(static_cast<int>(*(dest + 1)), 460);
}

TEST_F(LlvmLibcMBToWCTest, NullString) {
  wchar_t dest[2] = {L'O', L'K'};
  // reading on nullptr should return 0
  int n = LIBC_NAMESPACE::mbtowc(dest, nullptr, 2);
  ASSERT_EQ(n, 0);
  ASSERT_TRUE(dest[0] == L'O');
  // reading a null terminator should return 0
  const char *ch = "\0";
  n = LIBC_NAMESPACE::mbtowc(dest, ch, 1);
  ASSERT_EQ(n, 0);
}

TEST_F(LlvmLibcMBToWCTest, NullWCPtr) {
  const char ch[2] = {
      static_cast<char>(0xC2),
      static_cast<char>(0x8E),
  };
  // a null destination should still return the number of read bytes
  int n = LIBC_NAMESPACE::mbtowc(nullptr, ch, 2);
  ASSERT_EQ(n, 2);
}
