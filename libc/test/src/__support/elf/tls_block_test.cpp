//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for static TLS block allocation
///
//===----------------------------------------------------------------------===//

#include "src/__support/elf/elf_types.h"
#include "src/__support/elf/load_module.h"
#include "src/__support/elf/tls_block.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::elf::allocate_tls_block;
using LIBC_NAMESPACE::elf::free_tls_block;
using LIBC_NAMESPACE::elf::load_module;
using LIBC_NAMESPACE::elf::LoadedModule;
using LIBC_NAMESPACE::elf::Module;
using LIBC_NAMESPACE::elf::TLS_TCB_SIZE;
using LIBC_NAMESPACE::elf::TLS_VARIANT_2;
using LIBC_NAMESPACE::elf::TlsBlock;
using LIBC_NAMESPACE::elf::unmap_module;

namespace {
constexpr size_t PAGE = 4096;
constexpr const char *LIBC_PATH = LIBC_TEST_SHARED_OBJECT;
} // anonymous namespace

TEST(LlvmLibcElfTlsBlockTest, NoModulesMeansNoBlock) {
  auto result = allocate_tls_block(nullptr, 0);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value().storage, static_cast<void *>(nullptr));
  EXPECT_EQ(result.value().size, size_t(0));
}

TEST(LlvmLibcElfTlsBlockTest, ModuleWithoutTlsContributesNothing) {
  Module bare;
  auto result = allocate_tls_block(&bare, 1);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value().size, size_t(0));
}

TEST(LlvmLibcElfTlsBlockTest, CoversLibcOwnTlsSegment) {
  // This is the case that motivated the whole thing: libc.so has a PT_TLS
  // segment of its own, and a block sized from the executable alone has
  // nowhere to put it.
  auto loaded = load_module(LIBC_PATH, PAGE);
  ASSERT_TRUE(loaded.has_value());
  LoadedModule c = loaded.value();

  const ElfW(Phdr) *tls = c.module.tls();
  ASSERT_TRUE(tls != nullptr);
  EXPECT_GT(tls->p_memsz, LIBC_NAMESPACE::elf::Xword(0));

  Module modules[] = {c.module};
  auto result = allocate_tls_block(modules, 1);
  ASSERT_TRUE(result.has_value());
  TlsBlock block = result.value();

  ASSERT_TRUE(block.storage != nullptr);
  // The block has to hold libc's whole TLS segment plus the control block.
  EXPECT_GE(block.size, size_t(tls->p_memsz) + TLS_TCB_SIZE);

  // The thread pointer sits inside the allocation, at the top under variant 2
  // and at the bottom under variant 1.
  const uintptr_t base = reinterpret_cast<uintptr_t>(block.storage);
  EXPECT_GE(block.thread_pointer, base);
  EXPECT_LE(block.thread_pointer, base + block.size);
  if (TLS_VARIANT_2)
    EXPECT_EQ(block.thread_pointer, base + block.size - TLS_TCB_SIZE);
  else
    EXPECT_EQ(block.thread_pointer, base);

  free_tls_block(block);
  unmap_module(c.mapping);
}

TEST(LlvmLibcElfTlsBlockTest, InitialisedDataIsCopiedIn) {
  auto loaded = load_module(LIBC_PATH, PAGE);
  ASSERT_TRUE(loaded.has_value());
  LoadedModule c = loaded.value();
  const ElfW(Phdr) *tls = c.module.tls();
  ASSERT_TRUE(tls != nullptr);

  Module modules[] = {c.module};
  intptr_t offsets[1] = {0};
  auto result = allocate_tls_block(modules, 1, offsets);
  ASSERT_TRUE(result.has_value());
  TlsBlock block = result.value();

  if (tls->p_filesz != 0) {
    // The module's .tdata image must appear at the offset the allocator
    // reported, byte for byte.
    const unsigned char *image = reinterpret_cast<const unsigned char *>(
        c.module.load_bias() + tls->p_vaddr);
    const unsigned char *placed = reinterpret_cast<const unsigned char *>(
        static_cast<intptr_t>(block.thread_pointer) + offsets[0]);
    bool same = true;
    for (LIBC_NAMESPACE::elf::Xword i = 0; i < tls->p_filesz; ++i)
      if (placed[i] != image[i]) {
        same = false;
        break;
      }
    EXPECT_TRUE(same);
  }

  free_tls_block(block);
  unmap_module(c.mapping);
}
