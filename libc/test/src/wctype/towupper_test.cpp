//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for towupper.
///
//===----------------------------------------------------------------------===//

#include "hdr/wchar_macros.h" // for WEOF
#include "src/__support/wctype_utils.h"
#include "src/wctype/towupper.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcTowUpper, SimpleTest) {
  // ASCII Conversions
  EXPECT_EQ(LIBC_NAMESPACE::towupper(L'a'), static_cast<wint_t>(L'A'));
  EXPECT_EQ(LIBC_NAMESPACE::towupper(L'z'), static_cast<wint_t>(L'Z'));

  // ASCII Unchanged
  EXPECT_EQ(LIBC_NAMESPACE::towupper(L'A'), static_cast<wint_t>(L'A'));
  EXPECT_EQ(LIBC_NAMESPACE::towupper(L'Z'), static_cast<wint_t>(L'Z'));
  EXPECT_EQ(LIBC_NAMESPACE::towupper(L'1'), static_cast<wint_t>(L'1'));
  EXPECT_EQ(LIBC_NAMESPACE::towupper(L'\0'), static_cast<wint_t>(L'\0'));

  // WEOF Test
  EXPECT_EQ(LIBC_NAMESPACE::towupper(WEOF), static_cast<wint_t>(WEOF));

  // Boundary / Out-of-domain Tests (should return unchanged)
  EXPECT_EQ(LIBC_NAMESPACE::towupper(0xFFFF), static_cast<wint_t>(0xFFFF));
#if WCHAR_MAX > 0xFFFF
  EXPECT_EQ(LIBC_NAMESPACE::towupper(0x110000), static_cast<wint_t>(0x110000));
#endif

  // Non-ASCII Uppercase (Already uppercase: should remain unchanged
  // unconditionally in BOTH modes)
  EXPECT_EQ(LIBC_NAMESPACE::towupper(L'Α'), static_cast<wint_t>(L'Α'));
  EXPECT_EQ(LIBC_NAMESPACE::towupper(L'А'), static_cast<wint_t>(L'А'));
  EXPECT_EQ(LIBC_NAMESPACE::towupper(L'É'), static_cast<wint_t>(L'É'));
}

#if LIBC_CONF_WCTYPE_MODE == LIBC_WCTYPE_MODE_UTF8
TEST(LlvmLibcTowUpper, NonAsciiTestUtf8) {
  // Greek conversions
  EXPECT_EQ(LIBC_NAMESPACE::towupper(L'α'), static_cast<wint_t>(L'Α')); // alpha
  EXPECT_EQ(LIBC_NAMESPACE::towupper(L'ω'), static_cast<wint_t>(L'Ω')); // omega

  // Cyrillic conversions
  EXPECT_EQ(LIBC_NAMESPACE::towupper(L'а'), static_cast<wint_t>(L'А')); // A
  EXPECT_EQ(LIBC_NAMESPACE::towupper(L'я'), static_cast<wint_t>(L'Я')); // Ya

  // Accented Latin
  EXPECT_EQ(LIBC_NAMESPACE::towupper(L'é'), static_cast<wint_t>(L'É'));
  EXPECT_EQ(LIBC_NAMESPACE::towupper(L'ü'), static_cast<wint_t>(L'Ü'));

#if WCHAR_MAX > 0xFFFF
  // Deseret (Unicode Plane 1) conversions
  EXPECT_EQ(LIBC_NAMESPACE::towupper(L'\U00010428'),
            static_cast<wint_t>(L'\U00010400'));
#endif
}
#endif

#if LIBC_CONF_WCTYPE_MODE == LIBC_WCTYPE_MODE_ASCII
TEST(LlvmLibcTowUpper, NonAsciiTestAscii) {
  // Non-ASCII lowercase characters must return unchanged under ASCII-only mode
  EXPECT_EQ(LIBC_NAMESPACE::towupper(L'α'), static_cast<wint_t>(L'α'));
  EXPECT_EQ(LIBC_NAMESPACE::towupper(L'ω'), static_cast<wint_t>(L'ω'));

  EXPECT_EQ(LIBC_NAMESPACE::towupper(L'а'), static_cast<wint_t>(L'а'));
  EXPECT_EQ(LIBC_NAMESPACE::towupper(L'я'), static_cast<wint_t>(L'я'));

  EXPECT_EQ(LIBC_NAMESPACE::towupper(L'é'), static_cast<wint_t>(L'é'));
  EXPECT_EQ(LIBC_NAMESPACE::towupper(L'ü'), static_cast<wint_t>(L'ü'));

#if WCHAR_MAX > 0xFFFF
  EXPECT_EQ(LIBC_NAMESPACE::towupper(L'\U00010428'),
            static_cast<wint_t>(L'\U00010428'));
#endif
}
#endif
