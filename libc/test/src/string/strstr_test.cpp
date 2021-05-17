//===-- Unittests for strstr ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/strstr.h"
#include "utils/UnitTest/Test.h"

TEST(LlvmLibcStrStrTest, NeedleNotInHaystack) {
  const char *haystack = "12345";
  const char *needle = "a";
  ASSERT_STREQ(__llvm_libc::strstr(haystack, needle), nullptr);
}

TEST(LlvmLibcStrStrTest, NeedleIsEmptyString) {
  const char *haystack = "12345";
  const char *needle = "";
  ASSERT_STREQ(__llvm_libc::strstr(haystack, needle), haystack);
}

TEST(LlvmLibcStrStrTest, HaystackIsEmptyString) {
  const char *haystack = "";
  const char *needle = "12345";
  ASSERT_STREQ(__llvm_libc::strstr(haystack, needle), nullptr);
}

TEST(LlvmLibcStrStrTest, HaystackAndNeedleAreEmptyStrings) {
  const char *haystack = "";
  const char *needle = "";
  ASSERT_STREQ(__llvm_libc::strstr(haystack, needle), "");
}

TEST(LlvmLibcStrStrTest, HaystackAndNeedleAreSingleCharacters) {
  const char *haystack = "a";
  // Same characer returns that character.
  ASSERT_STREQ(__llvm_libc::strstr(haystack, /*needle=*/"a"), "a");
  // Different character returns nullptr.
  ASSERT_STREQ(__llvm_libc::strstr(haystack, /*needle=*/"b"), nullptr);
}

TEST(LlvmLibcStrStrTest, NeedleEqualToHaystack) {
  const char *haystack = "12345";
  const char *needle = "12345";
  ASSERT_STREQ(__llvm_libc::strstr(haystack, needle), "12345");
}

TEST(LlvmLibcStrStrTest, NeedleSmallerThanHaystack) {
  const char *haystack = "12345";
  const char *needle = "345";
  ASSERT_STREQ(__llvm_libc::strstr(haystack, needle), "345");
}

TEST(LlvmLibcStrStrTest, NeedleLargerThanHaystack) {
  const char *haystack = "123";
  const char *needle = "12345";
  ASSERT_STREQ(__llvm_libc::strstr(haystack, needle), nullptr);
}

TEST(LlvmLibcStrStrTest, NeedleAtBeginning) {
  const char *haystack = "12345";
  const char *needle = "12";
  ASSERT_STREQ(__llvm_libc::strstr(haystack, needle), "12345");
}

TEST(LlvmLibcStrStrTest, NeedleInMiddle) {
  const char *haystack = "abcdefghi";
  const char *needle = "def";
  ASSERT_STREQ(__llvm_libc::strstr(haystack, needle), "defghi");
}

TEST(LlvmLibcStrStrTest, NeedleDirectlyBeforeNullTerminator) {
  const char *haystack = "abcdefghi";
  const char *needle = "ghi";
  ASSERT_STREQ(__llvm_libc::strstr(haystack, needle), "ghi");
}

TEST(LlvmLibcStrStrTest, NeedlePastNullTerminator) {
  const char haystack[5] = {'1', '2', '\0', '3', '4'};
  // Shouldn't find anything after the null terminator.
  ASSERT_STREQ(__llvm_libc::strstr(haystack, /*needle=*/"3"), nullptr);
  ASSERT_STREQ(__llvm_libc::strstr(haystack, /*needle=*/"4"), nullptr);
}

TEST(LlvmLibcStrStrTest, PartialNeedle) {
  const char *haystack = "la_ap_lap";
  const char *needle = "lap";
  // Shouldn't find la or ap.
  ASSERT_STREQ(__llvm_libc::strstr(haystack, needle), "lap");
}

TEST(LlvmLibcStrStrTest, MisspelledNeedle) {
  const char *haystack = "atalloftwocities...wait, tale";
  const char *needle = "tale";
  ASSERT_STREQ(__llvm_libc::strstr(haystack, needle), "tale");
}

TEST(LlvmLibcStrStrTest, AnagramNeedle) {
  const char *haystack = "dgo_ogd_god_odg_gdo_dog";
  const char *needle = "dog";
  ASSERT_STREQ(__llvm_libc::strstr(haystack, needle), "dog");
}

TEST(LlvmLibcStrStrTest, MorphedNeedle) {
  // Changes a single letter in the needle to mismatch with the haystack.
  const char *haystack = "once upon a time";
  ASSERT_STREQ(__llvm_libc::strstr(haystack, "time"), "time");
  ASSERT_STREQ(__llvm_libc::strstr(haystack, "lime"), nullptr);
  ASSERT_STREQ(__llvm_libc::strstr(haystack, "tome"), nullptr);
  ASSERT_STREQ(__llvm_libc::strstr(haystack, "tire"), nullptr);
  ASSERT_STREQ(__llvm_libc::strstr(haystack, "timo"), nullptr);
}
