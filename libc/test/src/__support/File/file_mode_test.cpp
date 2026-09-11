//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// file_mode_test.cpp
/// This file contains possible test cases for FileMode class.
///
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

TEST(LlvmLibcFileModeTest, AllPossibleValidCombinations) {
  // This tracks all possible valid combinations

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
  EXPECT_TRUE(read_and_update.write_allowed());
  EXPECT_TRUE(read_and_update.read_allowed());

  // c. Read Binary
  constexpr FileMode read_binary("rb");
  EXPECT_TRUE(read_binary.is_valid());
  EXPECT_TRUE(read_binary.is_read());
  EXPECT_TRUE(read_binary.is_binary_format());
  EXPECT_TRUE(read_binary.read_allowed());

  // d. Read Update Binary
  constexpr FileMode read_update_binary("r+b");
  EXPECT_TRUE(read_update_binary.is_valid());
  EXPECT_TRUE(read_update_binary.is_read());
  EXPECT_TRUE(read_update_binary.is_update());
  EXPECT_TRUE(read_update_binary.is_binary_format());
  EXPECT_TRUE(read_update_binary.write_allowed());
  EXPECT_TRUE(read_update_binary.read_allowed());

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
  EXPECT_TRUE(write_and_update.read_allowed());

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

  // e. Write Binary Exclusive
  constexpr FileMode write_binary_exclusive("wbx");
  EXPECT_TRUE(write_binary_exclusive.is_valid());
  EXPECT_TRUE(write_binary_exclusive.is_write());
  EXPECT_TRUE(write_binary_exclusive.is_binary_format());
  EXPECT_TRUE(write_binary_exclusive.is_exclusive_create());
  EXPECT_TRUE(write_binary_exclusive.write_allowed());

  // f. Write Binary Update
  constexpr FileMode write_binary_update("wb+");
  EXPECT_TRUE(write_binary_update.is_valid());
  EXPECT_TRUE(write_binary_update.is_write());
  EXPECT_TRUE(write_binary_update.is_binary_format());
  EXPECT_TRUE(write_binary_update.is_update());
  EXPECT_TRUE(write_binary_update.write_allowed());
  EXPECT_TRUE(write_binary_update.read_allowed());

  // g. Write Update Exclusive
  constexpr FileMode write_update_exclusive("w+x");
  EXPECT_TRUE(write_update_exclusive.is_valid());
  EXPECT_TRUE(write_update_exclusive.is_write());
  EXPECT_TRUE(write_update_exclusive.is_update());
  EXPECT_TRUE(write_update_exclusive.is_exclusive_create());
  EXPECT_TRUE(write_update_exclusive.write_allowed());
  EXPECT_TRUE(write_update_exclusive.read_allowed());

  // h. Write Update Binary Exclusive
  constexpr FileMode write_update_binary_exclusive("w+bx");
  EXPECT_TRUE(write_update_binary_exclusive.is_valid());
  EXPECT_TRUE(write_update_binary_exclusive.is_write());
  EXPECT_TRUE(write_update_binary_exclusive.is_update());
  EXPECT_TRUE(write_update_binary_exclusive.is_binary_format());
  EXPECT_TRUE(write_update_binary_exclusive.is_exclusive_create());
  EXPECT_TRUE(write_update_binary_exclusive.write_allowed());
  EXPECT_TRUE(write_update_binary_exclusive.read_allowed());

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
  EXPECT_TRUE(append_and_update.read_allowed());

  // c. Append Binary
  constexpr FileMode append_binary("ab");
  EXPECT_TRUE(append_binary.is_valid());
  EXPECT_TRUE(append_binary.is_append());
  EXPECT_TRUE(append_binary.is_binary_format());
  EXPECT_TRUE(append_binary.write_allowed());

  // d. Append Update Binary
  constexpr FileMode append_update_binary("a+b");
  EXPECT_TRUE(append_update_binary.is_valid());
  EXPECT_TRUE(append_update_binary.is_append());
  EXPECT_TRUE(append_update_binary.is_update());
  EXPECT_TRUE(append_update_binary.is_binary_format());
  EXPECT_TRUE(append_update_binary.write_allowed());
  EXPECT_TRUE(append_update_binary.read_allowed());
}

TEST(LlvmLibcFileModeTest, InvalidCombinations) {
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
