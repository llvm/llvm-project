//===-- Unittests for CString ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for CString.
///
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/string.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/CPP/utility.h"
#include "src/__support/c_string.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::CString;
using LIBC_NAMESPACE::cpp::string;
using LIBC_NAMESPACE::cpp::string_view;

TEST(LlvmLibcCStringTest, DefaultConstructedEmpty) {
  const CString empty;
  EXPECT_STREQ(empty.c_str(), "");
  EXPECT_STREQ(static_cast<const char *>(empty), "");
  EXPECT_EQ(string_view(empty), string_view(""));
}

TEST(LlvmLibcCStringTest, ConstructFromShortAndLongStrings) {
  const CString from_cstr("short_path");
  EXPECT_STREQ(from_cstr.c_str(), "short_path");
  EXPECT_EQ(string_view(from_cstr), string_view("short_path"));

  static constexpr char LONG_PATH[] =
      "/tmp/llvm_libc_very_long_directory_prefix_exceeding_sso/test_file.dat";
  const CString from_string{string(LONG_PATH)};
  EXPECT_STREQ(from_string.c_str(), LONG_PATH);
  EXPECT_STREQ(static_cast<const char *>(from_string), LONG_PATH);
  EXPECT_EQ(string_view(from_string), string_view(LONG_PATH));
}

TEST(LlvmLibcCStringTest, MoveConstructAndAssign) {
  static constexpr char LONG_PATH[] =
      "/tmp/llvm_libc_move_semantics_heap_allocated_test_path/test_file.dat";

  CString original{string(LONG_PATH)};
  CString move_constructed(LIBC_NAMESPACE::cpp::move(original));
  EXPECT_STREQ(move_constructed.c_str(), LONG_PATH);

  CString move_assigned("initial");
  move_assigned = LIBC_NAMESPACE::cpp::move(move_constructed);
  EXPECT_STREQ(move_assigned.c_str(), LONG_PATH);
  EXPECT_EQ(string_view(move_assigned), string_view(LONG_PATH));
}
