//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains unit tests for wcwidth.
///
//===----------------------------------------------------------------------===//

#include "hdr/types/wchar_t.h"
#include "src/__support/wctype_utils.h"
#include "src/wchar/wcwidth.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcWcwidthTest, NullCharacter) {
  EXPECT_EQ(LIBC_NAMESPACE::wcwidth(L'\0'), 0);
}

TEST(LlvmLibcWcwidthTest, PrintableAscii) {
  EXPECT_EQ(LIBC_NAMESPACE::wcwidth(L' '), 1);
  EXPECT_EQ(LIBC_NAMESPACE::wcwidth(L'a'), 1);
  EXPECT_EQ(LIBC_NAMESPACE::wcwidth(L'5'), 1);
  EXPECT_EQ(LIBC_NAMESPACE::wcwidth(L'~'), 1);
}

TEST(LlvmLibcWcwidthTest, NonPrintableAsciiAndInvalid) {
  EXPECT_EQ(LIBC_NAMESPACE::wcwidth(L'\a'), -1);
  EXPECT_EQ(LIBC_NAMESPACE::wcwidth(L'\t'), -1);
  EXPECT_EQ(LIBC_NAMESPACE::wcwidth(L'\n'), -1);
  EXPECT_EQ(LIBC_NAMESPACE::wcwidth(L'\r'), -1);
  EXPECT_EQ(LIBC_NAMESPACE::wcwidth(static_cast<wchar_t>(0x7f)), -1);
  EXPECT_EQ(LIBC_NAMESPACE::wcwidth(static_cast<wchar_t>(0x80)), -1);
  EXPECT_EQ(LIBC_NAMESPACE::wcwidth(static_cast<wchar_t>(-1)), -1);
}

TEST(LlvmLibcWcwidthTest, NonAsciiCharacters) {
#if LIBC_CONF_WCTYPE_MODE == LIBC_WCTYPE_MODE_UTF8
  EXPECT_EQ(LIBC_NAMESPACE::wcwidth(L'\u00e1'), 1); // 'á'
  EXPECT_EQ(LIBC_NAMESPACE::wcwidth(L'\u03b2'), 1); // 'β'
#else
  EXPECT_EQ(LIBC_NAMESPACE::wcwidth(L'\u00e1'), -1);
  EXPECT_EQ(LIBC_NAMESPACE::wcwidth(L'\u03b2'), -1);
#endif // LIBC_CONF_WCTYPE_MODE == LIBC_WCTYPE_MODE_UTF8
}
