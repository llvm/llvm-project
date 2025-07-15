//===-- Unittests for wcpncpy --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/types/wchar_t.h"
#include "src/wchar/wcpncpy.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcWCPNCpyTest, EmptySrc) {
  // Empty src should lead to empty destination.
  wchar_t dest[4] = {L'a', L'b', L'c', L'\0'};
  const wchar_t *src = L"";
  LIBC_NAMESPACE::wcpncpy(dest, src, 3);
  ASSERT_TRUE(dest[0] == src[0]);
  ASSERT_TRUE(dest[0] == L'\0');
  // Rest should also be padded with L'\0'
  ASSERT_TRUE(dest[1] == L'\0');
  ASSERT_TRUE(dest[2] == L'\0');
}

TEST(LlvmLibcWCPNCpyTest, Untouched) {
  wchar_t dest[] = {L'a', L'b'};
  const wchar_t src[] = {L'x', L'\0'};
  LIBC_NAMESPACE::wcpncpy(dest, src, 0);
  ASSERT_TRUE(dest[0] == L'a');
  ASSERT_TRUE(dest[1] == L'b');
}

TEST(LlvmLibcWCPNCpyTest, CopyOne) {
  wchar_t dest[] = {L'a', L'b'};
  const wchar_t src[] = {L'x', L'y'};
  wchar_t *res = LIBC_NAMESPACE::wcpncpy(dest, src, 1);
  ASSERT_TRUE(dest[0] == L'x');
  ASSERT_TRUE(dest[1] == L'b');
  ASSERT_EQ(dest + 1, res);
}

TEST(LlvmLibcWCPNCpyTest, CopyNull) {
  wchar_t dest[] = {L'a', L'b'};
  const wchar_t src[] = {L'\0', L'y'};
  wchar_t *res = LIBC_NAMESPACE::wcpncpy(dest, src, 1);
  ASSERT_TRUE(dest[0] == L'\0');
  ASSERT_TRUE(dest[1] == L'b');
  ASSERT_EQ(dest, res);
}

TEST(LlvmLibcWCPNCpyTest, CopyPastSrc) {
  wchar_t dest[] = {L'a', L'b'};
  const wchar_t src[] = {L'\0', L'y'};
  wchar_t *res = LIBC_NAMESPACE::wcpncpy(dest, src, 2);
  ASSERT_TRUE(dest[0] == L'\0');
  ASSERT_TRUE(dest[1] == L'\0');
  ASSERT_EQ(dest, res);
}

TEST(LlvmLibcWCPNCpyTest, CopyTwoNoNull) {
  wchar_t dest[] = {L'a', L'b'};
  const wchar_t src[] = {L'x', L'y'};
  wchar_t *res = LIBC_NAMESPACE::wcpncpy(dest, src, 2);
  ASSERT_TRUE(dest[0] == L'x');
  ASSERT_TRUE(dest[1] == L'y');
  ASSERT_EQ(dest + 2, res);
}

TEST(LlvmLibcWCPNCpyTest, CopyTwoWithNull) {
  wchar_t dest[] = {L'a', L'b'};
  const wchar_t src[] = {L'x', L'\0'};
  wchar_t *res = LIBC_NAMESPACE::wcpncpy(dest, src, 2);
  ASSERT_TRUE(dest[0] == L'x');
  ASSERT_TRUE(dest[1] == L'\0');
  ASSERT_EQ(dest + 1, res);
}

TEST(LlvmLibcWCPNCpyTest, CopyAndFill) {
  wchar_t dest[] = {L'a', L'b', L'c'};
  wchar_t *res = LIBC_NAMESPACE::wcpncpy(dest, L"x", 3);
  ASSERT_TRUE(dest[0] == L'x');
  ASSERT_TRUE(dest[1] == L'\0');
  ASSERT_TRUE(dest[2] == L'\0');
  ASSERT_EQ(dest + 1, res);
}

#if defined(LIBC_ADD_NULL_CHECKS) && !defined(LIBC_HAS_SANITIZER)
TEST(LlvmLibcWCPNCpyTest, NullptrCrash) {
  // Passing in a nullptr should crash the program.
  EXPECT_DEATH([] { LIBC_NAMESPACE::wcpncpy(nullptr, nullptr, 1); },
               WITH_SIGNAL(-1));
}
#endif // LIBC_HAS_ADDRESS_SANITIZER
