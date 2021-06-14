//===-- strings_test.cpp ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "tests/scudo_unit_test.h"

#include "string_utils.h"

#include <limits.h>

TEST(ScudoStringsTest, Constructor) {
  scudo::ScopedString Str;
  EXPECT_EQ(0ul, Str.length());
  EXPECT_EQ('\0', *Str.data());
}

TEST(ScudoStringsTest, Basic) {
  scudo::ScopedString Str;
  Str.append("a%db%zdc%ue%zuf%xh%zxq%pe%sr", static_cast<int>(-1),
             static_cast<scudo::uptr>(-2), static_cast<unsigned>(-4),
             static_cast<scudo::uptr>(5), static_cast<unsigned>(10),
             static_cast<scudo::uptr>(11), reinterpret_cast<void *>(0x123),
             "_string_");
  EXPECT_EQ(Str.length(), strlen(Str.data()));

  std::string expectedString = "a-1b-2c4294967292e5fahbq0x";
  expectedString += std::string(SCUDO_POINTER_FORMAT_LENGTH - 3, '0');
  expectedString += "123e_string_r";
  EXPECT_EQ(Str.length(), strlen(Str.data()));
  EXPECT_STREQ(expectedString.c_str(), Str.data());
}

TEST(ScudoStringsTest, Clear) {
  scudo::ScopedString Str;
  Str.append("123");
  Str.clear();
  EXPECT_EQ(0ul, Str.length());
  EXPECT_EQ('\0', *Str.data());
}

TEST(ScudoStringsTest, ClearLarge) {
  scudo::ScopedString Str;
  for (int i = 0; i < 10000; ++i)
    Str.append("123");
  Str.clear();
  EXPECT_EQ(0ul, Str.length());
  EXPECT_EQ('\0', *Str.data());
}

TEST(ScudoStringsTest, Precision) {
  scudo::ScopedString Str;
  Str.append("%.*s", 3, "12345");
  EXPECT_EQ(Str.length(), strlen(Str.data()));
  EXPECT_STREQ("123", Str.data());
  Str.clear();
  Str.append("%.*s", 6, "12345");
  EXPECT_EQ(Str.length(), strlen(Str.data()));
  EXPECT_STREQ("12345", Str.data());
  Str.clear();
  Str.append("%-6s", "12345");
  EXPECT_EQ(Str.length(), strlen(Str.data()));
  EXPECT_STREQ("12345 ", Str.data());
}

static void fillString(scudo::ScopedString &Str, scudo::uptr Size) {
  for (scudo::uptr I = 0; I < Size; I++)
    Str.append("A");
}

TEST(ScudoStringTest, PotentialOverflows) {
  // Use a ScopedString that spans a page, and attempt to write past the end
  // of it with variations of append. The expectation is for nothing to crash.
  const scudo::uptr PageSize = scudo::getPageSizeCached();
  scudo::ScopedString Str;
  Str.clear();
  fillString(Str, 2 * PageSize);
  Str.clear();
  fillString(Str, PageSize - 64);
  Str.append("%-128s", "12345");
  Str.clear();
  fillString(Str, PageSize - 16);
  Str.append("%024x", 12345);
  Str.clear();
  fillString(Str, PageSize - 16);
  Str.append("EEEEEEEEEEEEEEEEEEEEEEEE");
}

template <typename T>
static void testAgainstLibc(const char *Format, T Arg1, T Arg2) {
  scudo::ScopedString Str;
  Str.append(Format, Arg1, Arg2);
  char Buffer[128];
  snprintf(Buffer, sizeof(Buffer), Format, Arg1, Arg2);
  EXPECT_EQ(Str.length(), strlen(Str.data()));
  EXPECT_STREQ(Buffer, Str.data());
}

TEST(ScudoStringsTest, MinMax) {
  testAgainstLibc<int>("%d-%d", INT_MIN, INT_MAX);
  testAgainstLibc<unsigned>("%u-%u", 0, UINT_MAX);
  testAgainstLibc<unsigned>("%x-%x", 0, UINT_MAX);
  testAgainstLibc<long>("%zd-%zd", LONG_MIN, LONG_MAX);
  testAgainstLibc<unsigned long>("%zu-%zu", 0, ULONG_MAX);
  testAgainstLibc<unsigned long>("%zx-%zx", 0, ULONG_MAX);
}

TEST(ScudoStringsTest, Padding) {
  testAgainstLibc<int>("%3d - %3d", 1, 0);
  testAgainstLibc<int>("%3d - %3d", -1, 123);
  testAgainstLibc<int>("%3d - %3d", -1, -123);
  testAgainstLibc<int>("%3d - %3d", 12, 1234);
  testAgainstLibc<int>("%3d - %3d", -12, -1234);
  testAgainstLibc<int>("%03d - %03d", 1, 0);
  testAgainstLibc<int>("%03d - %03d", -1, 123);
  testAgainstLibc<int>("%03d - %03d", -1, -123);
  testAgainstLibc<int>("%03d - %03d", 12, 1234);
  testAgainstLibc<int>("%03d - %03d", -12, -1234);
}
