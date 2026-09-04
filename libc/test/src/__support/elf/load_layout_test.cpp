//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for the PT_LOAD address space layout
///
//===----------------------------------------------------------------------===//

#include "src/__support/elf/elf_types.h"
#include "src/__support/elf/load_layout.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::elf::LoadLayout;

namespace {

constexpr size_t PAGE = 4096;

ElfW(Phdr)
    make(ElfW(Word) type, ElfW(Addr) vaddr, LIBC_NAMESPACE::elf::Xword memsz,
         LIBC_NAMESPACE::elf::Xword align) {
  ElfW(Phdr) p{};
  p.p_type = type;
  p.p_vaddr = vaddr;
  p.p_memsz = memsz;
  p.p_align = align;
  return p;
}

} // anonymous namespace

TEST(LlvmLibcElfLoadLayoutTest, EmptyWhenNoLoadSegments) {
  ElfW(Phdr) phdrs[] = {make(PT_DYNAMIC, 0x1000, 0x100, 8),
                        make(PT_TLS, 0x2000, 0x40, 8)};
  LoadLayout layout(phdrs, 2);
  EXPECT_TRUE(layout.empty());
  EXPECT_EQ(layout.segment_count(), size_t(0));
  EXPECT_EQ(layout.reservation_size(PAGE), size_t(0));
}

TEST(LlvmLibcElfLoadLayoutTest, SpansEverySegment) {
  // A typical shared object: text, rodata, then data, with other segment
  // types interleaved to make sure they are skipped.
  ElfW(Phdr) phdrs[] = {
      make(PT_PHDR, 0x40, 0x1c0, 8),       make(PT_LOAD, 0x0000, 0x2000, PAGE),
      make(PT_LOAD, 0x2000, 0x1000, PAGE), make(PT_DYNAMIC, 0x2f00, 0x100, 8),
      make(PT_LOAD, 0x4000, 0x0800, PAGE),
  };
  LoadLayout layout(phdrs, 5);
  EXPECT_FALSE(layout.empty());
  EXPECT_EQ(layout.segment_count(), size_t(3));
  EXPECT_EQ(layout.first_vaddr(), ElfW(Addr)(0x0000));
  EXPECT_EQ(layout.last_vaddr(), ElfW(Addr)(0x4800));
  // 0x0000 rounded down, 0x4800 rounded up to 0x5000.
  EXPECT_EQ(layout.reservation_size(PAGE), size_t(0x5000));
  EXPECT_EQ(layout.alignment(), LIBC_NAMESPACE::elf::Xword(PAGE));
}

TEST(LlvmLibcElfLoadLayoutTest, HandlesNonZeroBase) {
  // A non-PIE executable starts at a fixed address, so the reservation must
  // be measured from that base rather than from zero.
  ElfW(Phdr) phdrs[] = {
      make(PT_LOAD, 0x400000, 0x1000, PAGE),
      make(PT_LOAD, 0x402000, 0x1000, PAGE),
  };
  LoadLayout layout(phdrs, 2);
  EXPECT_EQ(layout.first_vaddr(), ElfW(Addr)(0x400000));
  EXPECT_EQ(layout.last_vaddr(), ElfW(Addr)(0x403000));
  EXPECT_EQ(layout.reservation_size(PAGE), size_t(0x3000));
}

TEST(LlvmLibcElfLoadLayoutTest, SegmentsOutOfOrder) {
  // Nothing requires PT_LOAD entries to be sorted, so the span must be taken
  // as a minimum and maximum rather than from the first and last entries.
  ElfW(Phdr) phdrs[] = {
      make(PT_LOAD, 0x4000, 0x1000, PAGE),
      make(PT_LOAD, 0x0000, 0x1000, PAGE),
  };
  LoadLayout layout(phdrs, 2);
  EXPECT_EQ(layout.first_vaddr(), ElfW(Addr)(0x0000));
  EXPECT_EQ(layout.last_vaddr(), ElfW(Addr)(0x5000));
  EXPECT_EQ(layout.reservation_size(PAGE), size_t(0x5000));
}

TEST(LlvmLibcElfLoadLayoutTest, UnalignedEndsAreRoundedUp) {
  ElfW(Phdr) phdrs[] = {make(PT_LOAD, 0x1001, 0x0001, PAGE)};
  LoadLayout layout(phdrs, 1);
  // Start page is 0x1000, end 0x1002 rounds up to 0x2000.
  EXPECT_EQ(layout.reservation_size(PAGE), size_t(0x1000));
}

TEST(LlvmLibcElfLoadLayoutTest, PageAlignHelpers) {
  EXPECT_EQ(LoadLayout::page_align_down(0x1fff, PAGE), ElfW(Addr)(0x1000));
  EXPECT_EQ(LoadLayout::page_align_down(0x1000, PAGE), ElfW(Addr)(0x1000));
  EXPECT_EQ(LoadLayout::page_align_up(0x1001, PAGE), ElfW(Addr)(0x2000));
  EXPECT_EQ(LoadLayout::page_align_up(0x1000, PAGE), ElfW(Addr)(0x1000));
}
