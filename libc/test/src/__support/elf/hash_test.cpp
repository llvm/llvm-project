//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for the ELF symbol hash functions
///
//===----------------------------------------------------------------------===//

#include "src/__support/elf/hash.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::elf::gnu_hash;
using LIBC_NAMESPACE::elf::sysv_hash;

TEST(LlvmLibcElfHashTest, SysVKnownValues) {
  EXPECT_EQ(sysv_hash(""), uint32_t(0x00000000));
  EXPECT_EQ(sysv_hash("printf"), uint32_t(0x077905a6));
  EXPECT_EQ(sysv_hash("malloc"), uint32_t(0x07383353));
  EXPECT_EQ(sysv_hash("_init"), uint32_t(0x00660504));
  EXPECT_EQ(sysv_hash("__libc_start_main"), uint32_t(0x0177ff8e));
}

TEST(LlvmLibcElfHashTest, SysVResultFitsIn28Bits) {
  // The top nibble is cleared every round, so no input can set it.
  EXPECT_EQ(sysv_hash("glibc_hash_collision_probe") & 0xf0000000u, uint32_t(0));
  EXPECT_EQ(sysv_hash("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa") & 0xf0000000u,
            uint32_t(0));
}

TEST(LlvmLibcElfHashTest, GnuKnownValues) {
  // The empty string yields the djb2 seed.
  EXPECT_EQ(gnu_hash(""), uint32_t(5381));
  EXPECT_EQ(gnu_hash("printf"), uint32_t(0x156b2bb8));
  EXPECT_EQ(gnu_hash("malloc"), uint32_t(0x0d39ad3d));
  EXPECT_EQ(gnu_hash("_init"), uint32_t(0x0ef18db8));
  EXPECT_EQ(gnu_hash("__libc_start_main"), uint32_t(0xf63d4e2e));
}

TEST(LlvmLibcElfHashTest, HashesAreUsableAtCompileTime) {
  static_assert(sysv_hash("printf") == 0x077905a6u);
  static_assert(gnu_hash("printf") == 0x156b2bb8u);
  EXPECT_TRUE(true);
}

TEST(LlvmLibcElfHashTest, HighBytesAreNotSignExtended) {
  // A char above 0x7f must be treated as unsigned.
  const char high[] = {static_cast<char>(0xff), '\0'};
  EXPECT_EQ(sysv_hash(high), uint32_t(0xff));
  EXPECT_EQ(gnu_hash(high), uint32_t(5381u * 33u + 0xffu));
}
