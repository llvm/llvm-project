//===-- Unittests for mbrtowc ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/types/mbstate_t.h"
#include "hdr/types/wchar_t.h"
#include "src/__support/wchar/mbstate.h"
#include "src/string/memset.h"
#include "src/wchar/mbrtowc.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/Test.h"

using LlvmLibcMBRToWCTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcMBRToWCTest, OneByte) {
  const char *ch = "A";
  wchar_t dest[2];
  // Testing if it works with nullptr mbstate_t
  mbstate_t *mb = nullptr;
  size_t n = LIBC_NAMESPACE::mbrtowc(dest, ch, 1, mb);
  ASSERT_EQ(static_cast<char>(*dest), 'A');
  ASSERT_EQ(static_cast<int>(n), 1);
  ASSERT_ERRNO_SUCCESS();

  // Should fail since we have not read enough
  n = LIBC_NAMESPACE::mbrtowc(dest, ch, 0, mb);
  ASSERT_EQ(static_cast<int>(n), -2);
  ASSERT_ERRNO_SUCCESS();
}

TEST_F(LlvmLibcMBRToWCTest, TwoByte) {
  const char ch[2] = {static_cast<char>(0xC2),
                      static_cast<char>(0x8E)}; //  car symbol
  wchar_t dest[2];
  mbstate_t mb;
  LIBC_NAMESPACE::memset(&mb, 0, sizeof(mbstate_t));
  size_t n = LIBC_NAMESPACE::mbrtowc(dest, ch, 2, &mb);
  ASSERT_EQ(static_cast<int>(*dest), 142);
  ASSERT_EQ(static_cast<int>(n), 2);
  ASSERT_ERRNO_SUCCESS();

  // Should fail since we have not read enough
  n = LIBC_NAMESPACE::mbrtowc(dest, ch, 1, &mb);
  ASSERT_EQ(static_cast<int>(n), -2);
  // Should pass after reading one more byte
  n = LIBC_NAMESPACE::mbrtowc(dest, ch + 1, 1, &mb);
  ASSERT_EQ(static_cast<int>(n), 1);
  ASSERT_EQ(static_cast<int>(*dest), 142);
  ASSERT_ERRNO_SUCCESS();
}

TEST_F(LlvmLibcMBRToWCTest, ThreeByte) {
  const char ch[3] = {static_cast<char>(0xE2), static_cast<char>(0x88),
                      static_cast<char>(0x91)}; // ∑ sigma symbol
  wchar_t dest[2];
  mbstate_t mb;
  LIBC_NAMESPACE::memset(&mb, 0, sizeof(mbstate_t));
  size_t n = LIBC_NAMESPACE::mbrtowc(dest, ch, 3, &mb);
  ASSERT_EQ(static_cast<int>(*dest), 8721);
  ASSERT_EQ(static_cast<int>(n), 3);
  ASSERT_ERRNO_SUCCESS();

  // Should fail since we have not read enough
  n = LIBC_NAMESPACE::mbrtowc(dest, ch, 1, &mb);
  ASSERT_EQ(static_cast<int>(n), -2);
  ASSERT_ERRNO_SUCCESS();
  // Should pass after reading two more bytes
  n = LIBC_NAMESPACE::mbrtowc(dest, ch + 1, 2, &mb);
  ASSERT_EQ(static_cast<int>(n), 2);
  ASSERT_EQ(static_cast<int>(*dest), 8721);
  ASSERT_ERRNO_SUCCESS();
}

TEST_F(LlvmLibcMBRToWCTest, FourByte) {
  const char ch[4] = {static_cast<char>(0xF0), static_cast<char>(0x9F),
                      static_cast<char>(0xA4),
                      static_cast<char>(0xA1)}; // 🤡 clown emoji
  wchar_t dest[2];
  mbstate_t mb;
  LIBC_NAMESPACE::memset(&mb, 0, sizeof(mbstate_t));
  size_t n = LIBC_NAMESPACE::mbrtowc(dest, ch, 4, &mb);
  ASSERT_EQ(static_cast<int>(*dest), 129313);
  ASSERT_EQ(static_cast<int>(n), 4);
  ASSERT_ERRNO_SUCCESS();
  // Should fail since we have not read enough
  n = LIBC_NAMESPACE::mbrtowc(dest, ch, 2, &mb);
  ASSERT_EQ(static_cast<int>(n), -2);
  ASSERT_ERRNO_SUCCESS();
  // Should pass after reading two more bytes
  n = LIBC_NAMESPACE::mbrtowc(dest, ch + 2, 2, &mb);
  ASSERT_EQ(static_cast<int>(n), 2);
  ASSERT_EQ(static_cast<int>(*dest), 129313);
  ASSERT_ERRNO_SUCCESS();
}

TEST_F(LlvmLibcMBRToWCTest, InvalidByte) {
  const char ch[1] = {static_cast<char>(0x80)};
  wchar_t dest[2];
  mbstate_t mb;
  LIBC_NAMESPACE::memset(&mb, 0, sizeof(mbstate_t));
  size_t n = LIBC_NAMESPACE::mbrtowc(dest, ch, 1, &mb);
  ASSERT_EQ(static_cast<int>(n), -1);
  ASSERT_ERRNO_EQ(EILSEQ);
}

TEST_F(LlvmLibcMBRToWCTest, InvalidMultiByte) {
  const char ch[4] = {static_cast<char>(0x80), static_cast<char>(0x00),
                      static_cast<char>(0x80),
                      static_cast<char>(0x00)}; // invalid sequence of bytes
  wchar_t dest[2];
  mbstate_t mb;
  LIBC_NAMESPACE::memset(&mb, 0, sizeof(mbstate_t));
  // Trying to push all 4 should error
  size_t n = LIBC_NAMESPACE::mbrtowc(dest, ch, 4, &mb);
  ASSERT_EQ(static_cast<int>(n), -1);
  ASSERT_ERRNO_EQ(EILSEQ);
  // Trying to push just the first one should error
  n = LIBC_NAMESPACE::mbrtowc(dest, ch, 1, &mb);
  ASSERT_EQ(static_cast<int>(n), -1);
  ASSERT_ERRNO_EQ(EILSEQ);
  // Trying to push the second and third should correspond to null wc
  n = LIBC_NAMESPACE::mbrtowc(dest, ch + 1, 2, &mb);
  ASSERT_EQ(static_cast<int>(n), 0);
  ASSERT_TRUE(*dest == L'\0');
  ASSERT_ERRNO_SUCCESS();
}

TEST_F(LlvmLibcMBRToWCTest, InvalidLastByte) {
  // Last byte is invalid since it does not have correct starting sequence.
  // 0xC0 --> 11000000 starting sequence should be 10xxxxxx
  const char ch[4] = {static_cast<char>(0xF1), static_cast<char>(0x80),
                      static_cast<char>(0x80), static_cast<char>(0xC0)};
  wchar_t dest[2];
  mbstate_t mb;
  LIBC_NAMESPACE::memset(&mb, 0, sizeof(mbstate_t));
  // Trying to push all 4 should error
  size_t n = LIBC_NAMESPACE::mbrtowc(dest, ch, 4, &mb);
  ASSERT_EQ(static_cast<int>(n), -1);
  ASSERT_ERRNO_EQ(EILSEQ);
}

TEST_F(LlvmLibcMBRToWCTest, ValidTwoByteWithExtraRead) {
  const char ch[3] = {static_cast<char>(0xC2), static_cast<char>(0x8E),
                      static_cast<char>(0x80)};
  wchar_t dest[2];
  mbstate_t mb;
  LIBC_NAMESPACE::memset(&mb, 0, sizeof(mbstate_t));
  // Trying to push all 3 should return valid 2 byte
  size_t n = LIBC_NAMESPACE::mbrtowc(dest, ch, 3, &mb);
  ASSERT_EQ(static_cast<int>(n), 2);
  ASSERT_EQ(static_cast<int>(*dest), 142);
  ASSERT_ERRNO_SUCCESS();
}

TEST_F(LlvmLibcMBRToWCTest, TwoValidTwoBytes) {
  const char ch[4] = {static_cast<char>(0xC2), static_cast<char>(0x8E),
                      static_cast<char>(0xC7), static_cast<char>(0x8C)};
  wchar_t dest[2];
  mbstate_t mb;
  LIBC_NAMESPACE::memset(&mb, 0, sizeof(mbstate_t));
  // mbstate should reset after reading first one
  size_t n = LIBC_NAMESPACE::mbrtowc(dest, ch, 2, &mb);
  ASSERT_EQ(static_cast<int>(n), 2);
  ASSERT_EQ(static_cast<int>(*dest), 142);
  ASSERT_ERRNO_SUCCESS();
  n = LIBC_NAMESPACE::mbrtowc(dest + 1, ch + 2, 2, &mb);
  ASSERT_EQ(static_cast<int>(n), 2);
  ASSERT_EQ(static_cast<int>(*(dest + 1)), 460);
  ASSERT_ERRNO_SUCCESS();
}

TEST_F(LlvmLibcMBRToWCTest, NullString) {
  wchar_t dest[2] = {L'O', L'K'};
  mbstate_t mb;
  LIBC_NAMESPACE::memset(&mb, 0, sizeof(mbstate_t));
  // reading on nullptr should return 0
  size_t n = LIBC_NAMESPACE::mbrtowc(dest, nullptr, 2, &mb);
  ASSERT_EQ(static_cast<int>(n), 0);
  ASSERT_TRUE(dest[0] == L'O');
  ASSERT_ERRNO_SUCCESS();
  // reading a null terminator should return 0
  const char *ch = "\0";
  n = LIBC_NAMESPACE::mbrtowc(dest, ch, 1, &mb);
  ASSERT_EQ(static_cast<int>(n), 0);
  ASSERT_ERRNO_SUCCESS();
}

TEST_F(LlvmLibcMBRToWCTest, NullDest) {
  const char ch[4] = {static_cast<char>(0xF0), static_cast<char>(0x9F),
                      static_cast<char>(0xA4),
                      static_cast<char>(0xA1)}; // 🤡 clown emoji
  mbstate_t mb;
  LIBC_NAMESPACE::memset(&mb, 0, sizeof(mbstate_t));
  // reading nullptr should return correct size
  size_t n = LIBC_NAMESPACE::mbrtowc(nullptr, ch, 10, &mb);
  ASSERT_EQ(static_cast<int>(n), 4);
  ASSERT_ERRNO_SUCCESS();
}

TEST_F(LlvmLibcMBRToWCTest, InvalidMBState) {
  const char ch[4] = {static_cast<char>(0xC2), static_cast<char>(0x8E),
                      static_cast<char>(0xC7), static_cast<char>(0x8C)};
  wchar_t dest[2] = {L'O', L'K'};
  mbstate_t *mb;
  LIBC_NAMESPACE::internal::mbstate inv;
  inv.total_bytes = 6;
  mb = reinterpret_cast<mbstate_t *>(&inv);
  // invalid mbstate should error
  size_t n = LIBC_NAMESPACE::mbrtowc(dest, ch, 2, mb);
  ASSERT_EQ(static_cast<int>(n), -1);
  ASSERT_ERRNO_EQ(EINVAL);
}
