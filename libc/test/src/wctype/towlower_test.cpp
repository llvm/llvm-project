//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for towlower.
///
//===----------------------------------------------------------------------===//

#include "hdr/wchar_macros.h" // for WEOF
#include "src/__support/wctype_utils.h"
#include "src/wctype/towlower.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcTowLower, SimpleTest) {
  // ASCII Conversions
  EXPECT_EQ(LIBC_NAMESPACE::towlower(L'A'), static_cast<wint_t>(L'a'));
  EXPECT_EQ(LIBC_NAMESPACE::towlower(L'Z'), static_cast<wint_t>(L'z'));

  // ASCII Unchanged
  EXPECT_EQ(LIBC_NAMESPACE::towlower(L'a'), static_cast<wint_t>(L'a'));
  EXPECT_EQ(LIBC_NAMESPACE::towlower(L'z'), static_cast<wint_t>(L'z'));
  EXPECT_EQ(LIBC_NAMESPACE::towlower(L'1'), static_cast<wint_t>(L'1'));
  EXPECT_EQ(LIBC_NAMESPACE::towlower(L'\0'), static_cast<wint_t>(L'\0'));

  // WEOF Test
  EXPECT_EQ(LIBC_NAMESPACE::towlower(WEOF), static_cast<wint_t>(WEOF));

  // Boundary / Out-of-domain Tests (should return unchanged)
  EXPECT_EQ(LIBC_NAMESPACE::towlower(0xFFFF), static_cast<wint_t>(0xFFFF));
#if WCHAR_MAX > 0xFFFF
  EXPECT_EQ(LIBC_NAMESPACE::towlower(0x110000), static_cast<wint_t>(0x110000));
#endif

  // Non-ASCII Lowercase (Already lowercase: should remain unchanged
  // unconditionally in BOTH modes)
  EXPECT_EQ(LIBC_NAMESPACE::towlower(L'α'), static_cast<wint_t>(L'α'));
  EXPECT_EQ(LIBC_NAMESPACE::towlower(L'а'), static_cast<wint_t>(L'а'));
  EXPECT_EQ(LIBC_NAMESPACE::towlower(L'é'), static_cast<wint_t>(L'é'));
}

#if LIBC_CONF_WCTYPE_MODE == LIBC_WCTYPE_MODE_UTF8
TEST(LlvmLibcTowLower, NonAsciiTestUtf8) {
  // Greek conversions
  EXPECT_EQ(LIBC_NAMESPACE::towlower(L'Α'), static_cast<wint_t>(L'α')); // alpha
  EXPECT_EQ(LIBC_NAMESPACE::towlower(L'Ω'), static_cast<wint_t>(L'ω')); // omega

  // Cyrillic conversions
  EXPECT_EQ(LIBC_NAMESPACE::towlower(L'А'), static_cast<wint_t>(L'а')); // A
  EXPECT_EQ(LIBC_NAMESPACE::towlower(L'Я'), static_cast<wint_t>(L'я')); // Ya

  // Accented Latin
  EXPECT_EQ(LIBC_NAMESPACE::towlower(L'É'), static_cast<wint_t>(L'é'));
  EXPECT_EQ(LIBC_NAMESPACE::towlower(L'Ü'), static_cast<wint_t>(L'ü'));

#if WCHAR_MAX > 0xFFFF
  // Deseret (Unicode Plane 1) conversions
  EXPECT_EQ(LIBC_NAMESPACE::towlower(L'\U00010400'),
            static_cast<wint_t>(L'\U00010428'));
#endif
}
#endif

#if LIBC_CONF_WCTYPE_MODE == LIBC_WCTYPE_MODE_ASCII
TEST(LlvmLibcTowLower, NonAsciiTestAscii) {
  // Non-ASCII uppercase characters must return unchanged under ASCII-only mode
  EXPECT_EQ(LIBC_NAMESPACE::towlower(L'Α'), static_cast<wint_t>(L'Α'));
  EXPECT_EQ(LIBC_NAMESPACE::towlower(L'Ω'), static_cast<wint_t>(L'Ω'));

  EXPECT_EQ(LIBC_NAMESPACE::towlower(L'А'), static_cast<wint_t>(L'А'));
  EXPECT_EQ(LIBC_NAMESPACE::towlower(L'Я'), static_cast<wint_t>(L'Я'));

  EXPECT_EQ(LIBC_NAMESPACE::towlower(L'É'), static_cast<wint_t>(L'É'));
  EXPECT_EQ(LIBC_NAMESPACE::towlower(L'Ü'), static_cast<wint_t>(L'Ü'));

#if WCHAR_MAX > 0xFFFF
  EXPECT_EQ(LIBC_NAMESPACE::towlower(L'\U00010400'),
            static_cast<wint_t>(L'\U00010400'));
#endif
}
#endif
