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
    const char *test_description;
    const char *mode;
    bool expects;
  };

  // testing for first character to be valid mode
  constexpr TestCase first_char_modes[] = {
      {.test_description = "valid append mode", .mode = "a", .expects = true},
      {.test_description = "valid read mode", .mode = "r", .expects = true},
      {.test_description = "valid write mode", .mode = "w", .expects = true},
      {.test_description = "update mode set as the first character",
       .mode = "+",
       .expects = false},
      {.test_description = "binary content set as the first character",
       .mode = "b",
       .expects = false},
      {.test_description = "exclusive create set as the first character",
       .mode = "x",
       .expects = false}};

  for (const TestCase &tc : first_char_modes) {
    const FileMode mode(tc.mode);
    EXPECT_EQ(mode.is_valid(), tc.expects);
  };
}

TEST(LlvmLibcFileModeTest, OnlyOneMainModeAllowed) {
  struct TestCase {
    const char *test_description;
    const char *mode;
    bool expects;
    const char *message = "";
  };

  // This tracks all possible valid combinations for a file mode with a main
  // mode both the ones that are allowed and the ones not allowed. The list is
  // exhaustive
  //
  // These are the valid main mode combinations
  //  read(r) = [update(+), binary(b)]
  //  write(w) = [update(+), binary(b), exclusive(x)]
  //  append(a) = [update(+), binary(b)]
  //
  // Invalid main mode combinations are
  //  read(r) = [write(w), append(a)]
  //  apppend(a) = [read(r), write(w)]
  constexpr TestCase modes_combination[] = {
      // read(r) = [update(+), binary(b)]
      {.test_description = "read only", .mode = "r", .expects = true},
      {.test_description = "read and update", .mode = "r+", .expects = true},
      {.test_description = "read binary", .mode = "rb", .expects = true},

      // write(w) = [update(+), binary(b), exclusive(x)]
      {.test_description = "write only", .mode = "w", .expects = true},
      {.test_description = "write and update", .mode = "w+", .expects = true},
      {.test_description = "write binary", .mode = "wb", .expects = true},
      {.test_description = "write exclusive", .mode = "wx", .expects = true},

      // append(a) = [update(+), binary(b)]
      {.test_description = "append and update", .mode = "a+", .expects = true},
      {.test_description = "append binary", .mode = "ab", .expects = true},

      // invalid main mode = read
      {
          .test_description = "read and write",
          .mode = "rw",
          .expects = false,
          .message = "read and write are both main modes and there can be only "
                     "one main mode",
      },
      {
          .test_description = "read and append",
          .mode = "ra",
          .expects = false,
          .message =
              "read and append are both main modes and there can be only "
              "one main mode",
      },

      // invalid main mode = write
      {
          .test_description = "write and read",
          .mode = "wr",
          .expects = false,
          .message = "write and read are all main modes and there can be only "
                     "one main mode",
      },
      {
          .test_description = "write and append",
          .mode = "wr",
          .expects = false,
          .message = "write and read are all main modes and there can be only "
                     "one main mode",
      },
      {
          .test_description = "append and read",
          .mode = "wr",
          .expects = false,
          .message = "append and read are all main modes and there can be only "
                     "one main mode",
      },
      {
          .test_description = "read,write and append",
          .mode = "rwa",
          .expects = false,
          .message =
              "read, write and append are all main modes and there can be only "
              "one main mode",
      },
  };

  for (const TestCase &tc : modes_combination) {
    const FileMode mode(tc.mode);
    EXPECT_EQ(mode.is_valid(), tc.expects) << tc.message;
  };
}

TEST(LlvmLibcFileModeTest, AllValidModes) {
  struct TestCase {
    const char *test_description;
    const char *mode;
    bool expects;
  };

  constexpr TestCase valid_modes[] = {
      // Read
      {.test_description = "read", .mode = "r", .expects = true},
      {.test_description = "read binary", .mode = "rb", .expects = true},
      {.test_description = "read update", .mode = "r+", .expects = true},
      {.test_description = "read update binary",
       .mode = "r+b",
       .expects = true},
      {.test_description = "read binary update",
       .mode = "rb+",
       .expects = true},

      // Write combinations
      {.test_description = "write", .mode = "w", .expects = true},
      {.test_description = "write binary", .mode = "wb", .expects = true},
      {.test_description = "write exclusive", .mode = "wx", .expects = true},
      {.test_description = "write binary exclusive",
       .mode = "wbx",
       .expects = true},
      {.test_description = "write update", .mode = "w+", .expects = true},
      {.test_description = "write update binary",
       .mode = "w+b",
       .expects = true},
      {.test_description = "write binary update",
       .mode = "wb+",
       .expects = true},
      {.test_description = "write update exclusive",
       .mode = "w+x",
       .expects = true},
      {.test_description = "write update binary exclusive",
       .mode = "w+bx",
       .expects = true},
      {.test_description = "write binary update exclusive",
       .mode = "wb+x",
       .expects = true},

      // Append combinations
      {.test_description = "append", .mode = "a", .expects = true},
      {.test_description = "append binary", .mode = "ab", .expects = true},
      {.test_description = "append update", .mode = "a+", .expects = true},
      {.test_description = "append update binary",
       .mode = "a+b",
       .expects = true},
      {.test_description = "append binary update",
       .mode = "ab+",
       .expects = true},
  };

  for (const TestCase &tc : valid_modes) {
    const FileMode mode(tc.mode);

    EXPECT_TRUE(mode.is_valid());
  }
}

TEST(LlvmLibcFileModeTest, WriteAllowedMode) {
  const FileMode mode("w+");

  EXPECT_TRUE(mode.is_valid());
  EXPECT_TRUE(mode.write_allowed());
}

TEST(LlvmLibcFileModeTest, FileModeIsReadAllowed) {
  const FileMode mode("r+");

  EXPECT_TRUE(mode.is_valid());
  EXPECT_TRUE(mode.read_allowed());
  EXPECT_TRUE(mode.is_update());
}

TEST(LlvmLibcFileModeTest, FileContentIsBinary) {
  struct TestCase {
    const char *mode;
  };

  constexpr TestCase binary_modes[] = {
      {.mode = "wb"},
      {.mode = "rb"},
      {.mode = "ab"},
  };

  for (const TestCase &tc : binary_modes) {
    const FileMode mode(tc.mode);

    EXPECT_TRUE(mode.is_valid());
    EXPECT_TRUE(mode.is_binary_format());
  }
}
