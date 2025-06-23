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

// TEST(LlvmLibcWCPCpyTest, EmptyDest) {
//   // Empty dest should result in src
//   const wchar_t *src = L"abc";
//   wchar_t dest[4];
//   wchar_t *result = LIBC_NAMESPACE::wcpcpy(dest, src);
//   ASSERT_EQ(dest + 3, result);
//   ASSERT_TRUE(result[0] == L'\0');
//   ASSERT_TRUE(dest[0] == L'a');
//   ASSERT_TRUE(dest[1] == L'b');
//   ASSERT_TRUE(dest[2] == L'c');
// }

// TEST(LlvmLibcWCPCpyTest, OffsetDest) {
//   // Offsetting should result in a concatenation.
//   const wchar_t *src = L"abc";
//   wchar_t dest[7];
//   dest[0] = L'x';
//   dest[1] = L'y';
//   dest[2] = L'z';
//   wchar_t *result = LIBC_NAMESPACE::wcpcpy(dest + 3, src);
//   ASSERT_TRUE(dest[0] == L'x');
//   ASSERT_TRUE(dest[1] == L'y');
//   ASSERT_TRUE(dest[2] == L'z');
//   ASSERT_TRUE(dest[3] == src[0]);
//   ASSERT_TRUE(dest[4] == src[1]);
//   ASSERT_TRUE(dest[5] == src[2]);
//   ASSERT_TRUE(result[0] == L'\0');
//   ASSERT_EQ(dest + 6, result);
// }

#if defined(LIBC_ADD_NULL_CHECKS) && !defined(LIBC_HAS_SANITIZER)
TEST(LlvmLibcWCPNCpyTest, NullptrCrash) {
  // Passing in a nullptr should crash the program.
  EXPECT_DEATH([] { LIBC_NAMESPACE::wcpncpy(nullptr, nullptr, 1); },
               WITH_SIGNAL(-1));
}
#endif // LIBC_HAS_ADDRESS_SANITIZER
