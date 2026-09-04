//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for the PT_DYNAMIC reader
///
//===----------------------------------------------------------------------===//

#include "src/__support/elf/dynamic.h"
#include "src/__support/elf/elf_types.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::elf::DynamicTable;

namespace {

// A synthetic dynamic array, so the test does not depend on how the test
// binary itself was linked.
constexpr ElfW(Addr) LOAD_BIAS = 0x400000;

// Aggregate initialised at namespace scope so there is no guard variable.
const ElfW(Dyn) SAMPLE[] = {
    {DT_NEEDED, {1}},
    {DT_SONAME, {17}},
    {DT_STRTAB, {0x1000}},
    {DT_SYMTAB, {0x2000}},
    {DT_NEEDED, {33}},
    {DT_RELASZ, {0x180}},
    {DT_NEEDED, {49}},
    {DT_NULL, {0}},
    // Anything past DT_NULL must not be seen.
    {DT_RPATH, {99}},
};

} // anonymous namespace

TEST(LlvmLibcElfDynamicTest, DefaultIsEmpty) {
  DynamicTable table;
  EXPECT_TRUE(table.empty());
  EXPECT_FALSE(table.value(DT_SONAME).has_value());
  EXPECT_FALSE(table.address(DT_STRTAB).has_value());
  EXPECT_FALSE(table.contains(DT_NEEDED));
  EXPECT_EQ(table.count(DT_NEEDED), size_t(0));
}

TEST(LlvmLibcElfDynamicTest, ValueIsNotBiased) {
  DynamicTable table(SAMPLE, LOAD_BIAS);
  ASSERT_TRUE(table.value(DT_RELASZ).has_value());
  EXPECT_EQ(*table.value(DT_RELASZ), LIBC_NAMESPACE::elf::Xword(0x180));
  ASSERT_TRUE(table.value(DT_SONAME).has_value());
  EXPECT_EQ(*table.value(DT_SONAME), LIBC_NAMESPACE::elf::Xword(17));
}

TEST(LlvmLibcElfDynamicTest, AddressIsBiased) {
  DynamicTable table(SAMPLE, LOAD_BIAS);
  ASSERT_TRUE(table.address(DT_STRTAB).has_value());
  EXPECT_EQ(*table.address(DT_STRTAB), ElfW(Addr)(LOAD_BIAS + 0x1000));
  ASSERT_TRUE(table.address(DT_SYMTAB).has_value());
  EXPECT_EQ(*table.address(DT_SYMTAB), ElfW(Addr)(LOAD_BIAS + 0x2000));
}

TEST(LlvmLibcElfDynamicTest, FindsFirstMatchOnly) {
  DynamicTable table(SAMPLE, LOAD_BIAS);
  ASSERT_TRUE(table.value(DT_NEEDED).has_value());
  EXPECT_EQ(*table.value(DT_NEEDED), LIBC_NAMESPACE::elf::Xword(1));
}

TEST(LlvmLibcElfDynamicTest, IteratesEveryMatch) {
  DynamicTable table(SAMPLE, LOAD_BIAS);
  EXPECT_EQ(table.count(DT_NEEDED), size_t(3));

  LIBC_NAMESPACE::elf::Xword total = 0;
  table.for_each(DT_NEEDED,
                 [&total](const ElfW(Dyn) &e) { total += e.d_un.d_val; });
  EXPECT_EQ(total, LIBC_NAMESPACE::elf::Xword(1 + 33 + 49));
}

TEST(LlvmLibcElfDynamicTest, StopsAtDtNull) {
  DynamicTable table(SAMPLE, LOAD_BIAS);
  EXPECT_FALSE(table.contains(DT_RPATH));
  EXPECT_EQ(table.count(DT_RPATH), size_t(0));
}

TEST(LlvmLibcElfDynamicTest, MissingTagsAreAbsent) {
  DynamicTable table(SAMPLE, LOAD_BIAS);
  EXPECT_FALSE(table.contains(DT_JMPREL));
  EXPECT_FALSE(table.value(DT_JMPREL).has_value());
  EXPECT_FALSE(table.address(DT_JMPREL).has_value());
}
