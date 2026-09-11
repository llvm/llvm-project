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
  // valid append mode
  constexpr FileMode append_only("a");
  EXPECT_TRUE(append_only.is_valid());
  EXPECT_TRUE(append_only.is_append());

  // valid read mode
  constexpr FileMode read_only("r");
  EXPECT_TRUE(read_only.is_valid());
  EXPECT_TRUE(read_only.is_read());

  // valid write mode
  constexpr FileMode write_only("w");
  EXPECT_TRUE(write_only.is_valid());
  EXPECT_TRUE(write_only.is_write());

  // update set as first character => invalid
  constexpr FileMode update_mode("+");
  EXPECT_FALSE(update_mode.is_valid());
  EXPECT_FALSE(update_mode.is_update());

  // binary bit flag set as first character => invalid
  constexpr FileMode binary_mode("b");
  EXPECT_FALSE(binary_mode.is_valid());
  EXPECT_FALSE(binary_mode.is_binary_format());

  // exclusive create set as first character => invalid
  constexpr FileMode exclusive_create("x");
  EXPECT_FALSE(exclusive_create.is_valid());
  EXPECT_FALSE(exclusive_create.is_exclusive_create());
}

TEST(LlvmLibcFileModeTest, OnlyOneMainModeAllowed) {
  // This tracks all possible valid combinations for a file mode with a main
  // mode both the ones that are allowed and the ones not allowed. The list
  // is exhaustive
  //
  // These are the valid main mode combinations
  //  read(r) = [update(+), binary(b)]
  //  write(w) = [update(+), binary(b), exclusive(x)]
  //  append(a) = [update(+), binary(b)]
  //
  // Invalid main mode combinations are
  //  read(r) = [write(w), append(a)]
  //  apppend(a) = [read(r), write(w)]

  // 1. Read: possible valid read combination modes

  // a. Read only
  constexpr FileMode readonly("r");
  EXPECT_TRUE(readonly.is_valid());
  EXPECT_TRUE(readonly.is_read());
  EXPECT_TRUE(readonly.read_allowed());

  // b. Read and Update mode
  constexpr FileMode read_and_update("r+");
  EXPECT_TRUE(read_and_update.is_valid());
  EXPECT_TRUE(read_and_update.is_read());
  EXPECT_TRUE(read_and_update.is_update());
  EXPECT_TRUE(read_and_update.read_allowed());

  // c. Read Binary
  constexpr FileMode read_binary("rb");
  EXPECT_TRUE(read_binary.is_valid());
  EXPECT_TRUE(read_binary.is_read());
  EXPECT_TRUE(read_binary.is_binary_format());
  EXPECT_TRUE(read_binary.read_allowed());

  // 2. Write: possible valid write combinations

  // a. Write only
  constexpr FileMode writeonly("w");
  EXPECT_TRUE(writeonly.is_valid());
  EXPECT_TRUE(writeonly.is_write());
  EXPECT_TRUE(writeonly.write_allowed());

  // b. Write and Update mode
  constexpr FileMode write_and_update("w+");
  EXPECT_TRUE(write_and_update.is_valid());
  EXPECT_TRUE(write_and_update.is_write());
  EXPECT_TRUE(write_and_update.is_update());
  EXPECT_TRUE(write_and_update.write_allowed());

  // c. Write Binary
  constexpr FileMode write_binary("wb");
  EXPECT_TRUE(write_binary.is_valid());
  EXPECT_TRUE(write_binary.is_write());
  EXPECT_TRUE(write_binary.is_binary_format());
  EXPECT_TRUE(write_binary.write_allowed());

  // d. Write Exclusive
  constexpr FileMode write_exclusive("wx");
  EXPECT_TRUE(write_exclusive.is_valid());
  EXPECT_TRUE(write_exclusive.is_write());
  EXPECT_TRUE(write_exclusive.is_exclusive_create());
  EXPECT_TRUE(write_exclusive.write_allowed());

  // 3. Append: possible valid append mode combinations

  // a. Append only
  constexpr FileMode appendonly("a");
  EXPECT_TRUE(appendonly.is_valid());
  EXPECT_TRUE(appendonly.is_append());
  EXPECT_TRUE(appendonly.write_allowed());

  // b. Append and Update
  constexpr FileMode append_and_update("a+");
  EXPECT_TRUE(append_and_update.is_valid());
  EXPECT_TRUE(append_and_update.is_append());
  EXPECT_TRUE(append_and_update.is_update());
  EXPECT_TRUE(append_and_update.write_allowed());

  // c. Append Binary
  constexpr FileMode append_binary("ab");
  EXPECT_TRUE(append_binary.is_valid());
  EXPECT_TRUE(append_binary.is_append());
  EXPECT_TRUE(append_binary.is_binary_format());
  EXPECT_TRUE(append_binary.write_allowed());

  // Invalid mode combinations

  // 1. Read and Write as main modes
  constexpr FileMode read_and_write("rw");
  EXPECT_FALSE(read_and_write.is_valid())
      << "read and write are both main modes and there can be only "
         "one main mode";

  // 2. Read and Append as main modes
  constexpr FileMode read_and_append("ra");
  EXPECT_FALSE(read_and_append.is_valid())
      << "read and append are both main modes and there can be only "
         "one main mode";

  // 3. Write and Append as main modes
  constexpr FileMode write_and_read("wr");
  EXPECT_FALSE(write_and_read.is_valid())
      << "write and read are both main modes and there can be only "
         "one main mode";

  // 4. Write and Append as main modes
  constexpr FileMode write_and_append("wa");
  EXPECT_FALSE(write_and_append.is_valid())
      << "write and append are both main modes and there can be only "
         "one main mode";

  // 5. Append and Read as main modes
  constexpr FileMode append_and_read("ar");
  EXPECT_FALSE(append_and_read.is_valid())
      << "append and read are both main modes and there can be only "
         "one main mode";

  // 6. Read, Write and Append as main modes
  constexpr FileMode read_write_append("rwa");
  EXPECT_FALSE(read_write_append.is_valid())
      << "read, write and append are both main modes and there can be only "
         "one main mode";
}

// TEST(LlvmLibcFileModeTest, AllValidModes) {
//   struct TestCase {
//     const char *test_description;
//     const char *mode;
//     bool expects;
//   };

//   constexpr TestCase valid_modes[] = {
//       // Read
//       {.test_description = "read", .mode = "r", .expects = true},
//       {.test_description = "read binary", .mode = "rb", .expects = true},
//       {.test_description = "read update", .mode = "r+", .expects = true},
//       {.test_description = "read update binary",
//        .mode = "r+b",
//        .expects = true},
//       {.test_description = "read binary update",
//        .mode = "rb+",
//        .expects = true},

//       // Write combinations
//       {.test_description = "write", .mode = "w", .expects = true},
//       {.test_description = "write binary", .mode = "wb", .expects = true},
//       {.test_description = "write exclusive", .mode = "wx", .expects = true},
//       {.test_description = "write binary exclusive",
//        .mode = "wbx",
//        .expects = true},
//       {.test_description = "write update", .mode = "w+", .expects = true},
//       {.test_description = "write update binary",
//        .mode = "w+b",
//        .expects = true},
//       {.test_description = "write binary update",
//        .mode = "wb+",
//        .expects = true},
//       {.test_description = "write update exclusive",
//        .mode = "w+x",
//        .expects = true},
//       {.test_description = "write update binary exclusive",
//        .mode = "w+bx",
//        .expects = true},
//       {.test_description = "write binary update exclusive",
//        .mode = "wb+x",
//        .expects = true},

//       // Append combinations
//       {.test_description = "append", .mode = "a", .expects = true},
//       {.test_description = "append binary", .mode = "ab", .expects = true},
//       {.test_description = "append update", .mode = "a+", .expects = true},
//       {.test_description = "append update binary",
//        .mode = "a+b",
//        .expects = true},
//       {.test_description = "append binary update",
//        .mode = "ab+",
//        .expects = true},
//   };

//   for (const TestCase &tc : valid_modes) {
//     const FileMode mode(tc.mode);

//     EXPECT_TRUE(mode.is_valid());
//   }
// }

// TEST(LlvmLibcFileModeTest, WriteAllowedMode) {
//   const FileMode mode("w+");

//   EXPECT_TRUE(mode.is_valid());
//   EXPECT_TRUE(mode.write_allowed());
// }

// TEST(LlvmLibcFileModeTest, FileModeIsReadAllowed) {
//   const FileMode mode("r+");

//   EXPECT_TRUE(mode.is_valid());
//   EXPECT_TRUE(mode.read_allowed());
//   EXPECT_TRUE(mode.is_update());
// }

// TEST(LlvmLibcFileModeTest, FileContentIsBinary) {
//   struct TestCase {
//     const char *mode;
//   };

//   constexpr TestCase binary_modes[] = {
//       {.mode = "wb"},
//       {.mode = "rb"},
//       {.mode = "ab"},
//   };

//   for (const TestCase &tc : binary_modes) {
//     const FileMode mode(tc.mode);

//     EXPECT_TRUE(mode.is_valid());
//     EXPECT_TRUE(mode.is_binary_format());
//   }
// }
