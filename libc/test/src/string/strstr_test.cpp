//===-- Unittests for strstr ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/strstr.h"
#include "utils/UnitTest/Test.h"

TEST(StrStrTest, NeedleNotInHaystack) {
  const char *haystack = "12345";
  const char *needle = "a";
  ASSERT_STREQ(__llvm_libc::strstr(haystack, needle), nullptr);
}

TEST(StrStrTest, NeedleIsEmptyString) {
  const char *haystack = "12345";
  const char *needle = "";
  ASSERT_STREQ(__llvm_libc::strstr(haystack, needle), haystack);
}

TEST(StrStrTest, HaystackIsEmptyString) {
  const char *haystack = "";
  const char *needle = "12345";
  ASSERT_STREQ(__llvm_libc::strstr(haystack, needle), nullptr);
}

TEST(StrStrTest, HaystackAndNeedleAreEmptyStrings) {
  const char *haystack = "";
  const char *needle = "";
  ASSERT_STREQ(__llvm_libc::strstr(haystack, needle), "");
}

TEST(StrStrTest, HaystackAndNeedleAreSingleCharacters) {
  const char *haystack = "a";
  // Same characer returns that character.
  ASSERT_STREQ(__llvm_libc::strstr(haystack, /*needle=*/"a"), "a");
  // Different character returns nullptr.
  ASSERT_STREQ(__llvm_libc::strstr(haystack, /*needle=*/"b"), nullptr);
}

TEST(StrStrTest, NeedleEqualToHaystack) {
  const char *haystack = "12345";
  const char *needle = "12345";
  ASSERT_STREQ(__llvm_libc::strstr(haystack, needle), "12345");
}

TEST(StrStrTest, NeedleSmallerThanHaystack) {
  const char *haystack = "12345";
  const char *needle = "345";
  ASSERT_STREQ(__llvm_libc::strstr(haystack, needle), "345");
}

TEST(StrStrTest, NeedleLargerThanHaystack) {
  const char *haystack = "123";
  const char *needle = "12345";
  ASSERT_STREQ(__llvm_libc::strstr(haystack, needle), nullptr);
}

TEST(StrStrTest, NeedleAtBeginning) {
  const char *haystack = "12345";
  const char *needle = "12";
  ASSERT_STREQ(__llvm_libc::strstr(haystack, needle), "12345");
}

TEST(StrStrTest, NeedleInMiddle) {
  const char *haystack = "abcdefghi";
  const char *needle = "def";
  ASSERT_STREQ(__llvm_libc::strstr(haystack, needle), "defghi");
}

TEST(StrStrTest, NeedleDirectlyBeforeNullTerminator) {
  const char *haystack = "abcdefghi";
  const char *needle = "ghi";
  ASSERT_STREQ(__llvm_libc::strstr(haystack, needle), "ghi");
}

TEST(StrStrTest, NeedlePastNullTerminator) {
  const char haystack[5] = {'1', '2', '\0', '3', '4'};
  // Shouldn't find anything after the null terminator.
  ASSERT_STREQ(__llvm_libc::strstr(haystack, /*needle=*/"3"), nullptr);
  ASSERT_STREQ(__llvm_libc::strstr(haystack, /*needle=*/"4"), nullptr);
}

TEST(StrStrTest, PartialNeedle) {
  const char *haystack = "la_ap_lap";
  const char *needle = "lap";
  // Shouldn't find la or ap.
  ASSERT_STREQ(__llvm_libc::strstr(haystack, needle), "lap");
}

TEST(StrStrTest, MisspelledNeedle) {
  const char *haystack = "atalloftwocities...wait, tale";
  const char *needle = "tale";
  ASSERT_STREQ(__llvm_libc::strstr(haystack, needle), "tale");
}

TEST(StrStrTest, AnagramNeedle) {
  const char *haystack = "dgo_ogd_god_odg_gdo_dog";
  const char *needle = "dog";
  ASSERT_STREQ(__llvm_libc::strstr(haystack, needle), "dog");
}

TEST(StrStrTest, MorphedNeedle) {
  // Changes a single letter in the needle to mismatch with the haystack.
  const char *haystack = "once upon a time";
  ASSERT_STREQ(__llvm_libc::strstr(haystack, "time"), "time");
  ASSERT_STREQ(__llvm_libc::strstr(haystack, "lime"), nullptr);
  ASSERT_STREQ(__llvm_libc::strstr(haystack, "tome"), nullptr);
  ASSERT_STREQ(__llvm_libc::strstr(haystack, "tire"), nullptr);
  ASSERT_STREQ(__llvm_libc::strstr(haystack, "timo"), nullptr);
}
