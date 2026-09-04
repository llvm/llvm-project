//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for the loaded module description
///
//===----------------------------------------------------------------------===//

#include "src/__support/elf/elf_types.h"
#include "src/__support/elf/module.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::elf::Module;

namespace {

const char STRTAB[] = "\0libtest.so.1\0libc.so.0\0libm.so.0";
// Offsets into STRTAB: 1 = libtest.so.1, 14 = libc.so.0, 24 = libm.so.0.

ElfW(Rela) RELAS[4];

const ElfW(Dyn) DYNAMIC[] = {
    {DT_NEEDED, {14}},
    {DT_NEEDED, {24}},
    {DT_SONAME, {1}},
    {DT_STRTAB, {0}}, // patched to &STRTAB at runtime
    {DT_RELA, {0}},   // patched to &RELAS at runtime
    {DT_RELASZ, {sizeof(RELAS)}},
    {DT_NULL, {0}},
};

// The dynamic array records addresses relative to the link address, and
// Module adds the load bias back on. Using a zero bias with absolute values
// keeps the arithmetic in the test trivial.
ElfW(Dyn) PATCHED[sizeof(DYNAMIC) / sizeof(DYNAMIC[0])];

ElfW(Phdr) PHDRS[2];

Module build() {
  for (size_t i = 0; i < sizeof(PATCHED) / sizeof(PATCHED[0]); ++i) {
    PATCHED[i] = DYNAMIC[i];
    if (PATCHED[i].d_tag == DT_STRTAB)
      PATCHED[i].d_un.d_ptr = reinterpret_cast<ElfW(Addr)>(STRTAB);
    if (PATCHED[i].d_tag == DT_RELA)
      PATCHED[i].d_un.d_ptr = reinterpret_cast<ElfW(Addr)>(RELAS);
  }

  PHDRS[0].p_type = PT_DYNAMIC;
  PHDRS[0].p_vaddr = reinterpret_cast<ElfW(Addr)>(PATCHED);
  PHDRS[1].p_type = PT_TLS;
  PHDRS[1].p_vaddr = 0x9000;
  PHDRS[1].p_memsz = 0x40;

  return Module(PHDRS, 2, /*load_bias=*/0, "libtest.so.1");
}

} // anonymous namespace

TEST(LlvmLibcElfModuleTest, DefaultIsInert) {
  Module m;
  EXPECT_TRUE(m.dynamic().empty());
  EXPECT_EQ(m.phnum(), ElfW(Half)(0));
  EXPECT_EQ(m.strtab(), static_cast<const char *>(nullptr));
  EXPECT_EQ(m.soname(), static_cast<const char *>(nullptr));
  EXPECT_EQ(m.tls(), static_cast<const ElfW(Phdr) *>(nullptr));
  EXPECT_TRUE(m.relocations().empty());
  EXPECT_TRUE(m.symbols().empty());
}

TEST(LlvmLibcElfModuleTest, FindsDynamicAndTlsSegments) {
  Module m = build();
  EXPECT_FALSE(m.dynamic().empty());
  ASSERT_TRUE(m.tls() != nullptr);
  EXPECT_EQ(m.tls()->p_memsz, LIBC_NAMESPACE::elf::Xword(0x40));
}

TEST(LlvmLibcElfModuleTest, ResolvesSoname) {
  Module m = build();
  ASSERT_TRUE(m.soname() != nullptr);
  EXPECT_STREQ(m.soname(), "libtest.so.1");
}

TEST(LlvmLibcElfModuleTest, ListsNeededInOrder) {
  Module m = build();
  const char *seen[4] = {};
  int count = 0;
  m.for_each_needed([&](const char *name) {
    if (count < 4)
      seen[count] = name;
    ++count;
  });
  ASSERT_EQ(count, 2);
  EXPECT_STREQ(seen[0], "libc.so.0");
  EXPECT_STREQ(seen[1], "libm.so.0");
}

TEST(LlvmLibcElfModuleTest, SizesRelocationTable) {
  Module m = build();
  EXPECT_EQ(m.relocations().size(), size_t(4));
  // No DT_JMPREL in this module.
  EXPECT_TRUE(m.plt_relocations().empty());
}

TEST(LlvmLibcElfModuleTest, NoGnuHashMeansNoSymbolTable) {
  // The sample carries no DT_GNU_HASH, so lookups must not be attempted.
  Module m = build();
  EXPECT_TRUE(m.symbols().empty());
  EXPECT_EQ(m.symbols().lookup("printf"),
            static_cast<const ElfW(Sym) *>(nullptr));
}
