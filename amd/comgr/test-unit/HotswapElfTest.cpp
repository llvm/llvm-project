//===- HotswapElfTest.cpp - Unit tests for HotSwap ELF layer --------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "comgr-hotswap-internal.h"
#include "gtest/gtest.h"
#include <cstring>

using namespace COMGR::hotswap;

// -- ElfView::create ----------------------------------------------------------

TEST(ElfView, RejectsTruncatedInput) {
  uint8_t Garbage[] = {0x7f, 'E', 'L', 'F', 0, 0, 0, 0};
  llvm::Expected<ElfView> ViewOrErr = ElfView::create(Garbage, sizeof(Garbage));
  EXPECT_FALSE((bool)ViewOrErr);
  llvm::consumeError(ViewOrErr.takeError());
}

TEST(ElfView, RejectsNonElfInput) {
  uint8_t NotElf[64] = {};
  llvm::Expected<ElfView> ViewOrErr = ElfView::create(NotElf, sizeof(NotElf));
  EXPECT_FALSE((bool)ViewOrErr);
  llvm::consumeError(ViewOrErr.takeError());
}

// -- ElfView::getKernelStaticLdsSize ------------------------------------------
//
// getKernelStaticLdsSize reads group_segment_fixed_size (the *static* LDS
// allocation; dynamic LDS is set by the host at dispatch time and not
// visible in the ELF) from a kernel descriptor symbol "<KernelName>.kd".
// Two unit tests cover the helper:
//   * negative path: no .kd symbol -> std::nullopt
//   * positive path: hand-crafted ELF with a .kd symbol pointing at an
//                    embedded kernel descriptor -> the embedded LDS size
// Real gfx1250 code-object coverage is added by the lit tests in #2302.

TEST(ElfView, GetKernelStaticLdsSizeReturnsNulloptWhenKdMissing) {
  // Build a minimal valid ELF64: header + .text + .shstrtab. ELFFile::create
  // succeeds, but no .kd symbol exists, so getKernelStaticLdsSize must take
  // the missing-KD branch.
  using namespace llvm::ELF;
  static constexpr size_t BufSize = 512;
  alignas(8) uint8_t Buf[BufSize] = {};

  static constexpr uint64_t ShOff = sizeof(Elf64_Ehdr);
  static constexpr uint64_t StrTabOff = 256;
  static constexpr uint64_t TextOff = 320;
  static constexpr uint64_t TextSize = 16;

  const char StrTab[] = "\0.text\0.shstrtab\0";
  std::memcpy(Buf + StrTabOff, StrTab, sizeof(StrTab));

  Elf64_Ehdr Ehdr{};
  Ehdr.e_ident[0] = 0x7f;
  Ehdr.e_ident[1] = 'E';
  Ehdr.e_ident[2] = 'L';
  Ehdr.e_ident[3] = 'F';
  Ehdr.e_ident[EI_CLASS] = ELFCLASS64;
  Ehdr.e_ident[EI_DATA] = ELFDATA2LSB;
  Ehdr.e_ident[EI_VERSION] = EV_CURRENT;
  Ehdr.e_ident[EI_OSABI] = ELFOSABI_AMDGPU_HSA;
  Ehdr.e_type = ET_REL;
  Ehdr.e_machine = EM_AMDGPU;
  Ehdr.e_version = EV_CURRENT;
  Ehdr.e_shoff = ShOff;
  Ehdr.e_ehsize = sizeof(Elf64_Ehdr);
  Ehdr.e_shentsize = sizeof(Elf64_Shdr);
  Ehdr.e_shnum = 3;
  Ehdr.e_shstrndx = 2;
  std::memcpy(Buf, &Ehdr, sizeof(Ehdr));

  // Shdr[1] = .text
  Elf64_Shdr Sh1{};
  Sh1.sh_name = 1;
  Sh1.sh_type = SHT_PROGBITS;
  Sh1.sh_flags = SHF_ALLOC | SHF_EXECINSTR;
  Sh1.sh_offset = TextOff;
  Sh1.sh_size = TextSize;
  std::memcpy(Buf + ShOff + 1 * sizeof(Elf64_Shdr), &Sh1, sizeof(Sh1));

  // Shdr[2] = .shstrtab
  Elf64_Shdr Sh2{};
  Sh2.sh_name = 7;
  Sh2.sh_type = SHT_STRTAB;
  Sh2.sh_offset = StrTabOff;
  Sh2.sh_size = sizeof(StrTab);
  std::memcpy(Buf + ShOff + 2 * sizeof(Elf64_Shdr), &Sh2, sizeof(Sh2));

  llvm::Expected<ElfView> ViewOrErr = ElfView::create(Buf, BufSize);
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());
  EXPECT_EQ(ViewOrErr->getKernelStaticLdsSize("nonexistent_kernel"),
            std::nullopt);
}

TEST(ElfView, GetKernelStaticLdsSizeReadsLdsSizeFromKernelDescriptor) {
  // Build a minimal AMDGPU ELF64 with the section topology that
  // findKernelDescriptor walks: 6 sections (NULL, .text, .rodata, .strtab,
  // .symtab, .shstrtab). The kernel descriptor is embedded at the start of
  // .rodata with a known group_segment_fixed_size value, and a symbol named
  // "test_kernel.kd" in .symtab points at it. getKernelStaticLdsSize must
  // return the embedded static-LDS size unchanged.
  using namespace llvm::ELF;
  static constexpr size_t BufSize = 1024;
  alignas(8) uint8_t Buf[BufSize] = {};

  // Section file offsets and sizes. Layout choices keep each section
  // 8-byte aligned so the ELF parser is happy.
  static constexpr uint64_t ShOff = sizeof(Elf64_Ehdr);
  static constexpr uint64_t TextOff = 0x1C0;
  static constexpr uint64_t TextSize = 16;
  static constexpr uint64_t RodataOff = 0x1D0;
  static constexpr uint64_t KdSize = 64;
  static constexpr uint64_t StrTabOff = 0x210;
  static constexpr uint64_t SymTabOff = 0x220;
  static constexpr uint64_t ShStrTabOff = 0x250;
  static constexpr uint64_t SymCount = 2;
  static constexpr uint32_t TestLdsSize = 16384;

  // Section name string table. Entries: "" .text .rodata .strtab .symtab
  // .shstrtab. Offsets pinned in the shdr fields below.
  const char ShStrTab[] = "\0.text\0.rodata\0.strtab\0.symtab\0.shstrtab\0";
  std::memcpy(Buf + ShStrTabOff, ShStrTab, sizeof(ShStrTab));

  // Symbol name string table. Single named symbol "test_kernel.kd" at
  // offset 1; offset 0 is the conventional empty name.
  const char StrTab[] = "\0test_kernel.kd\0";
  std::memcpy(Buf + StrTabOff, StrTab, sizeof(StrTab));

  Elf64_Ehdr Ehdr{};
  Ehdr.e_ident[0] = 0x7f;
  Ehdr.e_ident[1] = 'E';
  Ehdr.e_ident[2] = 'L';
  Ehdr.e_ident[3] = 'F';
  Ehdr.e_ident[EI_CLASS] = ELFCLASS64;
  Ehdr.e_ident[EI_DATA] = ELFDATA2LSB;
  Ehdr.e_ident[EI_VERSION] = EV_CURRENT;
  Ehdr.e_ident[EI_OSABI] = ELFOSABI_AMDGPU_HSA;
  Ehdr.e_type = ET_REL;
  Ehdr.e_machine = EM_AMDGPU;
  Ehdr.e_version = EV_CURRENT;
  Ehdr.e_shoff = ShOff;
  Ehdr.e_ehsize = sizeof(Elf64_Ehdr);
  Ehdr.e_shentsize = sizeof(Elf64_Shdr);
  Ehdr.e_shnum = 6;
  Ehdr.e_shstrndx = 5;
  std::memcpy(Buf, &Ehdr, sizeof(Ehdr));

  // Section header table. Shdr[0] is the conventional NULL section (left
  // as the buffer's zero-init). Each non-null shdr is zero-initialized by
  // Elf64_Shdr{} so unspecified fields (sh_addr, sh_info, sh_addralign,
  // ...) are explicitly zero.

  // Shdr[1] = .text
  Elf64_Shdr Sh1{};
  Sh1.sh_name = 1;
  Sh1.sh_type = SHT_PROGBITS;
  Sh1.sh_flags = SHF_ALLOC | SHF_EXECINSTR;
  Sh1.sh_offset = TextOff;
  Sh1.sh_size = TextSize;
  std::memcpy(Buf + ShOff + 1 * sizeof(Elf64_Shdr), &Sh1, sizeof(Sh1));

  // Shdr[2] = .rodata (holds the kernel descriptor)
  Elf64_Shdr Sh2{};
  Sh2.sh_name = 7;
  Sh2.sh_type = SHT_PROGBITS;
  Sh2.sh_flags = SHF_ALLOC;
  Sh2.sh_offset = RodataOff;
  Sh2.sh_size = KdSize;
  std::memcpy(Buf + ShOff + 2 * sizeof(Elf64_Shdr), &Sh2, sizeof(Sh2));

  // Shdr[3] = .strtab (symbol names)
  Elf64_Shdr Sh3{};
  Sh3.sh_name = 15;
  Sh3.sh_type = SHT_STRTAB;
  Sh3.sh_offset = StrTabOff;
  Sh3.sh_size = sizeof(StrTab);
  std::memcpy(Buf + ShOff + 3 * sizeof(Elf64_Shdr), &Sh3, sizeof(Sh3));

  // Shdr[4] = .symtab; sh_link = 3 (.strtab)
  Elf64_Shdr Sh4{};
  Sh4.sh_name = 23;
  Sh4.sh_type = SHT_SYMTAB;
  Sh4.sh_offset = SymTabOff;
  Sh4.sh_size = sizeof(Elf64_Sym) * SymCount;
  Sh4.sh_link = 3;
  Sh4.sh_entsize = sizeof(Elf64_Sym);
  std::memcpy(Buf + ShOff + 4 * sizeof(Elf64_Shdr), &Sh4, sizeof(Sh4));

  // Shdr[5] = .shstrtab (section names)
  Elf64_Shdr Sh5{};
  Sh5.sh_name = 31;
  Sh5.sh_type = SHT_STRTAB;
  Sh5.sh_offset = ShStrTabOff;
  Sh5.sh_size = sizeof(ShStrTab);
  std::memcpy(Buf + ShOff + 5 * sizeof(Elf64_Shdr), &Sh5, sizeof(Sh5));

  // Kernel descriptor body: group_segment_fixed_size at offset 0. The rest
  // of the 64-byte descriptor stays zero, which is fine for a read-only
  // helper that only consumes one field.
  std::memcpy(Buf + RodataOff, &TestLdsSize, sizeof(TestLdsSize));

  // Symbol table. Slot 0 is the conventional null symbol (left as the
  // buffer's zero-init). Slot 1 names "test_kernel.kd" at .strtab offset 1
  // and points at the start of .rodata (st_value=0).
  Elf64_Sym Sym1{};
  Sym1.st_name = 1;
  Sym1.setBindingAndType(STB_GLOBAL, STT_OBJECT);
  Sym1.st_shndx = 2;
  Sym1.st_size = KdSize;
  std::memcpy(Buf + SymTabOff + 1 * sizeof(Elf64_Sym), &Sym1, sizeof(Sym1));

  llvm::Expected<ElfView> ViewOrErr = ElfView::create(Buf, BufSize);
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());
  std::optional<uint32_t> Lds =
      ViewOrErr->getKernelStaticLdsSize("test_kernel");
  ASSERT_TRUE(Lds.has_value());
  EXPECT_EQ(*Lds, TestLdsSize);
}
