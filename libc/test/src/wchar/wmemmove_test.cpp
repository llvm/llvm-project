//===-- Unittests for wmemmove --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/types/size_t.h"
#include "hdr/types/wchar_t.h"
#include "src/wchar/wmemmove.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcWMemmoveTest, MoveZeroByte) {
  wchar_t buffer[] = {L'a', L'b', L'y', L'z'};

  wchar_t *ret = LIBC_NAMESPACE::wmemmove(buffer, buffer + 2, 0);
  EXPECT_EQ(ret, buffer);

  const wchar_t expected[] = {L'a', L'b', L'y', L'z'};
  EXPECT_TRUE(buffer[0] == expected[0]);
  EXPECT_TRUE(buffer[1] == expected[1]);
  EXPECT_TRUE(buffer[2] == expected[2]);
  EXPECT_TRUE(buffer[3] == expected[3]);
}

TEST(LlvmLibcWMemmoveTest, DstAndSrcPointToSameAddress) {
  wchar_t buffer[] = {L'a', L'b'};

  wchar_t *ret = LIBC_NAMESPACE::wmemmove(buffer, buffer, 1);
  EXPECT_EQ(ret, buffer);

  const wchar_t expected[] = {L'a', L'b'};
  EXPECT_TRUE(buffer[0] == expected[0]);
  EXPECT_TRUE(buffer[1] == expected[1]);
}

TEST(LlvmLibcWMemmoveTest, DstStartsBeforeSrc) {
  // Set boundary at beginning and end for not overstepping when
  // copy forward or backward.
  wchar_t buffer[] = {L'z', L'a', L'b', L'c', L'z'};

  wchar_t *dst = buffer + 1;
  wchar_t *ret = LIBC_NAMESPACE::wmemmove(dst, buffer + 2, 2);
  EXPECT_EQ(ret, dst);

  const wchar_t expected[] = {L'z', L'b', L'c', L'c', L'z'};
  EXPECT_TRUE(buffer[0] == expected[0]);
  EXPECT_TRUE(buffer[1] == expected[1]);
  EXPECT_TRUE(buffer[2] == expected[2]);
  EXPECT_TRUE(buffer[3] == expected[3]);
  EXPECT_TRUE(buffer[4] == expected[4]);
}

TEST(LlvmLibcWMemmoveTest, DstStartsAfterSrc) {
  wchar_t buffer[] = {L'z', L'a', L'b', L'c', L'z'};

  wchar_t *dst = buffer + 2;
  wchar_t *ret = LIBC_NAMESPACE::wmemmove(dst, buffer + 1, 2);
  EXPECT_EQ(ret, dst);

  const wchar_t expected[] = {L'z', L'a', L'a', L'b', L'z'};
  EXPECT_TRUE(buffer[0] == expected[0]);
  EXPECT_TRUE(buffer[1] == expected[1]);
  EXPECT_TRUE(buffer[2] == expected[2]);
  EXPECT_TRUE(buffer[3] == expected[3]);
  EXPECT_TRUE(buffer[4] == expected[4]);
}

// e.g. `Dst` follow `src`.
// str: [abcdefghij]
//      [__src_____]
//      [_____Dst__]
TEST(LlvmLibcWMemmoveTest, SrcFollowDst) {
  wchar_t buffer[] = {L'z', L'a', L'b', L'z'};

  wchar_t *dst = buffer + 1;
  wchar_t *ret = LIBC_NAMESPACE::wmemmove(dst, buffer + 2, 1);
  EXPECT_EQ(ret, dst);

  const char expected[] = {L'z', L'b', L'b', L'z'};
  EXPECT_TRUE(buffer[0] == expected[0]);
  EXPECT_TRUE(buffer[1] == expected[1]);
  EXPECT_TRUE(buffer[2] == expected[2]);
  EXPECT_TRUE(buffer[3] == expected[3]);
}

TEST(LlvmLibcWMemmoveTest, DstFollowSrc) {
  wchar_t buffer[] = {L'z', L'a', L'b', L'z'};

  wchar_t *dst = buffer + 2;
  wchar_t *ret = LIBC_NAMESPACE::wmemmove(dst, buffer + 1, 1);
  EXPECT_EQ(ret, dst);

  const char expected[] = {L'z', L'a', L'a', L'z'};
  EXPECT_TRUE(buffer[0] == expected[0]);
  EXPECT_TRUE(buffer[1] == expected[1]);
  EXPECT_TRUE(buffer[2] == expected[2]);
  EXPECT_TRUE(buffer[3] == expected[3]);
}

#if defined(LIBC_ADD_NULL_CHECKS) && !defined(LIBC_HAS_SANITIZER)
TEST(LlvmLibcWMemmoveTest, NullptrCrash) {
  wchar_t buffer[] = {L'a', L'b'};
  // Passing in a nullptr should crash the program.
  EXPECT_DEATH([&buffer] { LIBC_NAMESPACE::wmemmove(buffer, nullptr, 2); },
               WITH_SIGNAL(-1));
  EXPECT_DEATH([&buffer] { LIBC_NAMESPACE::wmemmove(nullptr, buffer, 2); },
               WITH_SIGNAL(-1));
}
#endif // LIBC_HAS_ADDRESS_SANITIZER
