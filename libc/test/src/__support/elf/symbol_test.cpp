//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for GNU hash symbol lookup
///
//===----------------------------------------------------------------------===//

#include "src/__support/elf/symbol.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::elf::GnuSymbolTable;

namespace {

// A DT_GNU_HASH table laid out the way a linker emits one: header, Bloom
// filter, buckets, then the hash array. Index 0 of .dynsym is the reserved
// null entry, so symoffset is 1 and the hashed symbols are sorted by bucket.
//
// Symbols, in bucket order: strlen, printf, malloc, free, fopen.
struct Table {
  uint32_t nbuckets = 3;
  uint32_t symoffset = 1;
  uint32_t bloom_size = 2;
  uint32_t bloom_shift = 6;
  // Every bit set, so that nothing is turned away before the buckets are
  // read. The filter is as wide as an address, which differs between the ELF
  // classes, and a real linker's bits for one would be wrong for the other.
  ElfW(Addr) bloom[2] = { ~ElfW(Addr)(0), ~ElfW(Addr)(0) };
  uint32_t buckets[3] = {1, 2, 0};
  uint32_t chain[5] = {0x1c93bb9d, 0x156b2bb8, 0x0d39ad3c, 0x7c96f086,
                       0x0f738b7d};
};

const Table TABLE;

// Offsets into STRTAB below.
const char STRTAB[] = "\0strlen\0printf\0malloc\0free\0fopen";

ElfW(Sym) make_sym(uint32_t name_offset, ElfW(Addr) value) {
  ElfW(Sym) s{};
  s.st_name = name_offset;
  s.st_value = value;
  s.st_info = ELF64_ST_INFO(STB_GLOBAL, STT_FUNC);
  s.st_shndx = 1;
  return s;
}

const ElfW(Sym) SYMTAB[] = {
    make_sym(0, 0),       // reserved null entry
    make_sym(1, 0x1000),  // strlen
    make_sym(8, 0x2000),  // printf
    make_sym(15, 0x3000), // malloc
    make_sym(22, 0x4000), // free
    make_sym(27, 0x5000), // fopen
};

GnuSymbolTable table() { return GnuSymbolTable(&TABLE, SYMTAB, STRTAB); }

} // anonymous namespace

TEST(LlvmLibcElfSymbolTest, DefaultIsEmpty) {
  GnuSymbolTable empty;
  EXPECT_TRUE(empty.empty());
  EXPECT_EQ(empty.lookup("printf"), static_cast<const ElfW(Sym) *>(nullptr));
}

TEST(LlvmLibcElfSymbolTest, FindsEverySymbol) {
  GnuSymbolTable t = table();
  ASSERT_FALSE(t.empty());
  // Each name must resolve to its own entry, including the ones that share a
  // bucket and are only separated by walking the chain.
  ASSERT_TRUE(t.lookup("strlen") != nullptr);
  EXPECT_EQ(t.lookup("strlen")->st_value, ElfW(Addr)(0x1000));
  ASSERT_TRUE(t.lookup("printf") != nullptr);
  EXPECT_EQ(t.lookup("printf")->st_value, ElfW(Addr)(0x2000));
  ASSERT_TRUE(t.lookup("malloc") != nullptr);
  EXPECT_EQ(t.lookup("malloc")->st_value, ElfW(Addr)(0x3000));
  ASSERT_TRUE(t.lookup("free") != nullptr);
  EXPECT_EQ(t.lookup("free")->st_value, ElfW(Addr)(0x4000));
  ASSERT_TRUE(t.lookup("fopen") != nullptr);
  EXPECT_EQ(t.lookup("fopen")->st_value, ElfW(Addr)(0x5000));
}

TEST(LlvmLibcElfSymbolTest, MissingNamesReturnNull) {
  GnuSymbolTable t = table();
  EXPECT_EQ(t.lookup("calloc"), static_cast<const ElfW(Sym) *>(nullptr));
  EXPECT_EQ(t.lookup(""), static_cast<const ElfW(Sym) *>(nullptr));
  EXPECT_EQ(t.lookup("qsort"), static_cast<const ElfW(Sym) *>(nullptr));
}

TEST(LlvmLibcElfSymbolTest, PrefixesDoNotMatch) {
  // A chain walk that compared only the hash, or compared with a prefix match,
  // would wrongly resolve these.
  GnuSymbolTable t = table();
  EXPECT_EQ(t.lookup("printf_"), static_cast<const ElfW(Sym) *>(nullptr));
  EXPECT_EQ(t.lookup("print"), static_cast<const ElfW(Sym) *>(nullptr));
  EXPECT_EQ(t.lookup("fre"), static_cast<const ElfW(Sym) *>(nullptr));
}
