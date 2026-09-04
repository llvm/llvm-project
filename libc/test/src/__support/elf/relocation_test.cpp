//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for relative relocation processing
///
//===----------------------------------------------------------------------===//

#include "src/__support/elf/elf_types.h"
#include "src/__support/elf/relocation.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::elf::RelaTable;
using LIBC_NAMESPACE::elf::RELATIVE_RELOC;

namespace {

// The targets a relocation writes into. They are addressed as an offset from
// a pretend load bias, so the test mirrors how a real module is laid out.
ElfW(Addr) SLOTS[4];

ElfW(Addr) bias() {
  // r_offset is added to the bias, so treat SLOTS as if it sat at offset 0.
  return reinterpret_cast<ElfW(Addr)>(&SLOTS[0]);
}

ElfW(Rela)
    make(ElfW(Addr) offset, uint32_t type, LIBC_NAMESPACE::elf::Sxword addend) {
  ElfW(Rela) r{};
  r.r_offset = offset;
#ifdef __LP64__
  r.r_info = ELF64_R_INFO(LIBC_NAMESPACE::elf::Xword(0),
                          LIBC_NAMESPACE::elf::Xword(type));
#else
  r.r_info = ELF32_R_INFO(0, type);
#endif
  r.r_addend = addend;
  return r;
}

} // anonymous namespace

TEST(LlvmLibcElfRelocationTest, EmptyTable) {
  RelaTable table;
  EXPECT_TRUE(table.empty());
  EXPECT_EQ(table.size(), size_t(0));
  EXPECT_EQ(table.apply_relative(0x1000), size_t(0));
}

TEST(LlvmLibcElfRelocationTest, SizeComesFromByteCount) {
  ElfW(Rela) entries[3] = {};
  RelaTable table(entries, sizeof(entries));
  EXPECT_EQ(table.size(), size_t(3));
  EXPECT_FALSE(table.empty());
  EXPECT_EQ(table.end() - table.begin(), ptrdiff_t(3));
}

TEST(LlvmLibcElfRelocationTest, AppliesRelativeRelocations) {
  for (auto &slot : SLOTS)
    slot = 0xdeadbeef;

  const ElfW(Addr) load_bias = bias();
  ElfW(Rela) entries[] = {
      make(0 * sizeof(ElfW(Addr)), RELATIVE_RELOC, 0x10),
      make(1 * sizeof(ElfW(Addr)), RELATIVE_RELOC, 0x20),
      make(3 * sizeof(ElfW(Addr)), RELATIVE_RELOC, 0x30),
  };

  RelaTable table(entries, sizeof(entries));
  EXPECT_EQ(table.apply_relative(load_bias), size_t(3));

  EXPECT_EQ(SLOTS[0], ElfW(Addr)(load_bias + 0x10));
  EXPECT_EQ(SLOTS[1], ElfW(Addr)(load_bias + 0x20));
  // Slot 2 had no relocation and must be untouched.
  EXPECT_EQ(SLOTS[2], ElfW(Addr)(0xdeadbeef));
  EXPECT_EQ(SLOTS[3], ElfW(Addr)(load_bias + 0x30));
}

TEST(LlvmLibcElfRelocationTest, IgnoresOtherRelocationTypes) {
  for (auto &slot : SLOTS)
    slot = 0xdeadbeef;

  // A type that is deliberately not the relative one. Applying it here would
  // be wrong: it needs a symbol table that does not exist yet.
  constexpr uint32_t SOME_OTHER_TYPE = RELATIVE_RELOC + 1;
  ElfW(Rela) entries[] = {
      make(0 * sizeof(ElfW(Addr)), SOME_OTHER_TYPE, 0x10),
      make(1 * sizeof(ElfW(Addr)), RELATIVE_RELOC, 0x20),
  };

  RelaTable table(entries, sizeof(entries));
  EXPECT_EQ(table.apply_relative(bias()), size_t(1));
  EXPECT_EQ(SLOTS[0], ElfW(Addr)(0xdeadbeef));
  EXPECT_EQ(SLOTS[1], ElfW(Addr)(bias() + 0x20));
}

TEST(LlvmLibcElfRelocationTest, NegativeAddendsWork) {
  for (auto &slot : SLOTS)
    slot = 0;
  const ElfW(Addr) load_bias = bias();
  ElfW(Rela) entries[] = {make(0, RELATIVE_RELOC, -16)};
  RelaTable table(entries, sizeof(entries));
  EXPECT_EQ(table.apply_relative(load_bias), size_t(1));
  EXPECT_EQ(SLOTS[0], ElfW(Addr)(load_bias - 16));
}
