//===-- Unittests for file mode class//---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/File/file_mode.h"
#include "src/__support/macros/config.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::FileMode;

TEST(LlvmLibcFileModeTest, FirstCharacterMustBeAValidMode) {
  // creates a table structure to group tests
  struct TestCase {
    const char *test_name;
    const char *mode;
    bool expects;
  };

  constexpr TestCase valid_modes[] = {
      {.test_name = "valid append mode", .mode = "a", .expects = true},
      {.test_name = "valid read mode", .mode = "r", .expects = true},
      {.test_name = "valid write mode", .mode = "w", .expects = true},
  };

  for (const TestCase &tc : valid_modes) {
    const FileMode mode(tc.mode);
    EXPECT_EQ(mode.is_valid(), tc.expects);
  };

  constexpr TestCase invalid_first_char_modes[] = {
      {.test_name = "update mode set as the first character",
       .mode = "+",
       .expects = false},
      {.test_name = "binary content set as the first character",
       .mode = "b",
       .expects = false},
      {.test_name = "exclusive create set as the first character",
       .mode = "x",
       .expects = false},
  };

  for (const TestCase &tc : invalid_first_char_modes) {
    const FileMode mode(tc.mode);
    EXPECT_EQ(mode.is_valid(), tc.expects);
  };
}

// Test(LlvmLibcFileModeTest, OnlyOneMainModeAllowed) {}
//
// Test(LlvmLibcFileModeTest, WriteAllowedMode) {}
//
// Test(LlvmLibcFileModeTest, FileModeIsReadAllowed) {}
//
// Test(LlvmLibcFileModeTest, WriteOnlyAllowed) {}
//
// Test(LlvmLibcFileModeTest, ReadOnlyAllowed) {}
//
// Test(LlvmLibcFileModeTest, AppendAllowed) {}
//
// Test(LlvmLibcFileModeTest, BinaryContentBitIsSet) {}
//
// Test(LlvmLibcFileModeTest, ExclusiveCreateBitIsSet) {}
