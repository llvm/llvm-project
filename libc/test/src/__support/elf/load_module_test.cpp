//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for loading a shared object
///
//===----------------------------------------------------------------------===//

#include "src/__support/elf/load_module.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::elf::load_module;
using LIBC_NAMESPACE::elf::LoadedModule;
using LIBC_NAMESPACE::elf::unmap_module;

namespace {
constexpr size_t PAGE = 4096;
// Set by the build to the libc.so this test was built alongside.
constexpr const char *SO_PATH = LIBC_TEST_SHARED_OBJECT;
} // anonymous namespace

TEST(LlvmLibcElfLoadModuleTest, RejectsMissingFile) {
  auto result = load_module("testdata/definitely-not-here.so", PAGE);
  EXPECT_FALSE(result.has_value());
}

TEST(LlvmLibcElfLoadModuleTest, RejectsNonElfFile) {
  // The test binary's own source tree always has this.
  auto result = load_module("/etc/hostname", PAGE);
  EXPECT_FALSE(result.has_value());
}

TEST(LlvmLibcElfLoadModuleTest, LoadsRealSharedObject) {
  auto result = load_module(SO_PATH, PAGE);
  ASSERT_TRUE(result.has_value());
  LoadedModule loaded = result.value();

  ASSERT_TRUE(loaded.mapping.reservation != nullptr);
  EXPECT_GT(loaded.mapping.reservation_size, size_t(0));

  // The ELF header landed at the load bias.
  const unsigned char *base =
      reinterpret_cast<const unsigned char *>(loaded.mapping.load_bias);
  EXPECT_EQ(base[0], static_cast<unsigned char>(0x7f));
  EXPECT_EQ(base[1], static_cast<unsigned char>('E'));
  EXPECT_EQ(base[2], static_cast<unsigned char>('L'));
  EXPECT_EQ(base[3], static_cast<unsigned char>('F'));

  // A real shared object has a dynamic section and a string table.
  EXPECT_FALSE(loaded.module.dynamic().empty());
  EXPECT_TRUE(loaded.module.strtab() != nullptr);

  unmap_module(loaded.mapping);
}

TEST(LlvmLibcElfLoadModuleTest, ResolvesSymbolsInLoadedObject) {
  auto result = load_module(SO_PATH, PAGE);
  ASSERT_TRUE(result.has_value());
  LoadedModule loaded = result.value();

  auto symbols = loaded.module.symbols();
  ASSERT_FALSE(symbols.empty());

  // These are exported by libc.so, and looking them up exercises the whole
  // chain: mapping, dynamic section parsing, and GNU hash lookup.
  const ElfW(Sym) *printf_sym = symbols.lookup("printf");
  ASSERT_TRUE(printf_sym != nullptr);
  EXPECT_NE(printf_sym->st_value, ElfW(Addr)(0));

  EXPECT_TRUE(symbols.lookup("memcpy") != nullptr);
  EXPECT_TRUE(symbols.lookup("fopen") != nullptr);
  // Something that is definitely not there.
  EXPECT_TRUE(symbols.lookup("definitely_not_a_libc_symbol") == nullptr);

  unmap_module(loaded.mapping);
}

TEST(LlvmLibcElfLoadModuleTest, ReportsSoname) {
  auto result = load_module(SO_PATH, PAGE);
  ASSERT_TRUE(result.has_value());
  LoadedModule loaded = result.value();
  // libc.so is built with a SONAME, so it must come back non-null.
  EXPECT_TRUE(loaded.module.soname() != nullptr);
  unmap_module(loaded.mapping);
}
