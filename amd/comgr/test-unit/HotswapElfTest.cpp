//===- HotswapElfTest.cpp - Unit tests for HotSwap ELF layer --------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "comgr-hotswap-internal.h"
#include "comgr-test-elf-utils.h"
#include "gtest/gtest.h"

#include <cstring>
#include <limits>

using namespace COMGR::hotswap;

static std::vector<uint8_t> makeText(size_t Size = 16) {
  return std::vector<uint8_t>(Size, 0);
}

static unsigned readReservedSgprs(const std::vector<uint8_t> &Bytes,
                                  uint64_t KernelDescriptorOffset) {
  namespace hsa = llvm::amdhsa;

  uint32_t Rsrc1 = 0;
  std::memcpy(&Rsrc1,
              Bytes.data() + KernelDescriptorOffset +
                  offsetof(hsa::kernel_descriptor_t, compute_pgm_rsrc1),
              sizeof(Rsrc1));
  return (AMDHSA_BITS_GET(
              Rsrc1, hsa::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT) +
          1) *
         8;
}

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

// -- ElfView::findKernelAtAddress ---------------------------------------------

TEST(ElfView, FindKernelAtAddressResolvesNearestPrecedingForZeroSizeSymbol) {
  // AMDGPU kernel entry symbols frequently have st_size == 0 (the size lives on
  // the .kd object symbol), so an exact [st_value, st_value + st_size)
  // containment test never matches. The lookup must resolve via the
  // nearest-preceding STT_FUNC symbol instead.
  comgr_test::KernelDescriptorElfOptions Opts;
  Opts.KernelName = "zero_size_kernel";
  Opts.TextAddr = 0x1000;
  Opts.ZeroSizeKernelSym = true;
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(makeText(), Opts);

  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());

  // findKernelAtAddress takes a virtual address; at the entry and at an
  // interior offset the zero-size symbol still resolves.
  EXPECT_EQ(ViewOrErr->findKernelAtAddress(0x1000), "zero_size_kernel");
  EXPECT_EQ(ViewOrErr->findKernelAtAddress(0x1000 + 4), "zero_size_kernel");
  // An address before the symbol has no preceding function symbol to resolve.
  EXPECT_EQ(ViewOrErr->findKernelAtAddress(0x0FF0), "");
}

// -- findNearestSled ----------------------------------------------------------

TEST(FindNearestSled, SkipsSledsOutsideInstructionFunctionRange) {
  std::vector<NopSled> Sleds;
  // {Start, End, WritePos, FunctionStart, FunctionEnd}
  Sleds.push_back({/*Start=*/0, /*End=*/32, /*WritePos=*/0,
                   /*FunctionStart=*/0, /*FunctionEnd=*/32});
  Sleds.push_back({/*Start=*/96, /*End=*/128, /*WritePos=*/96,
                   /*FunctionStart=*/96, /*FunctionEnd=*/160});

  NopSled *Sled = findNearestSled(Sleds, 108, 8);
  ASSERT_NE(Sled, nullptr);
  EXPECT_EQ(Sled->Start, 96u);

  EXPECT_EQ(findNearestSled(Sleds, 64, 8), nullptr);
}

// -- ElfView::getKernelStaticLdsSize ------------------------------------------

TEST(ElfView, GetKernelStaticLdsSizeReturnsNulloptWhenKdMissing) {
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(makeText());

  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());
  EXPECT_EQ(ViewOrErr->getKernelStaticLdsSize("nonexistent_kernel"),
            std::nullopt);
}

TEST(ElfView, GetKernelStaticLdsSizeReadsLdsSizeFromKernelDescriptor) {
  static constexpr uint32_t TestLdsSize = 16384;

  comgr_test::KernelDescriptorElfOptions Opts;
  Opts.ElfType = llvm::ELF::ET_REL;
  Opts.KernelName = "test_kernel";
  Opts.TextAddr = 0;
  Opts.RodataAddr = 0;
  Opts.GroupSegmentFixedSize = TestLdsSize;
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(makeText(), Opts);

  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());
  std::optional<uint32_t> Lds =
      ViewOrErr->getKernelStaticLdsSize("test_kernel");
  ASSERT_TRUE(Lds.has_value());
  EXPECT_EQ(*Lds, TestLdsSize);
}

TEST(ElfView, KernelDescriptorsEnumeratesAndUpdatesEntryOffset) {
  namespace hsa = llvm::amdhsa;

  comgr_test::KernelDescriptorElfOptions Opts;
  Opts.KernelName = "entry_kernel";
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(makeText(), Opts);

  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());
  llvm::ArrayRef<KernelDescriptorInfo> KDs = ViewOrErr->kernelDescriptors();
  ASSERT_EQ(KDs.size(), 1u);
  EXPECT_EQ(KDs[0].KernelName, "entry_kernel");
  EXPECT_EQ(KDs[0].VAddr, Obj.RodataAddr);
  EXPECT_EQ(KDs[0].EntryOffset, Obj.EntryOffset);
  EXPECT_EQ(ViewOrErr->getKernelDescriptorVAddr("entry_kernel"),
            Obj.RodataAddr);

  const int64_t NewOff = -128;
  ASSERT_TRUE(
      ViewOrErr->updateKernelDescriptorEntryOffset("entry_kernel", NewOff));
  int64_t ReadBack = 0;
  std::memcpy(
      &ReadBack,
      Obj.Bytes.data() + Obj.KernelDescriptorOffset +
          offsetof(hsa::kernel_descriptor_t, kernel_code_entry_byte_offset),
      sizeof(ReadBack));
  EXPECT_EQ(ReadBack, NewOff);
  ASSERT_EQ(ViewOrErr->kernelDescriptors().size(), 1u);
  EXPECT_EQ(ViewOrErr->kernelDescriptors()[0].EntryOffset, NewOff);

  // Prime the descriptor fallback cache before changing the encoded count.
  EXPECT_EQ(ViewOrErr->getKernelSgprCount("entry_kernel"), 8u);
  ASSERT_TRUE(ViewOrErr->updateKernelDescriptorSgprCount("entry_kernel", 10));
  EXPECT_GE(readReservedSgprs(Obj.Bytes, Obj.KernelDescriptorOffset), 10u);
  EXPECT_EQ(ViewOrErr->getKernelSgprCount("entry_kernel"), 16u);
}

TEST(ElfView, KernelDescriptorsSkipsKdWhenFileOffsetOverflows) {
  comgr_test::KernelDescriptorElfOptions Opts;
  Opts.KernelName = "overflow_kernel";
  Opts.RodataAddr = 0x1000;
  Opts.KernelDescriptorSymbolValue =
      std::numeric_limits<uint64_t>::max() - 0x20;
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(makeText(), Opts);

  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());
  EXPECT_TRUE(ViewOrErr->kernelDescriptors().empty());
  EXPECT_EQ(ViewOrErr->findKernelDescriptor("overflow_kernel"), nullptr);
}

// growWithTrampolines appends the pool at a fresh high virtual address instead
// of growing .text and shifting everything after it, so existing allocatable
// symbols keep their addresses. (This replaces the earlier test that pinned the
// buggy shifting behavior; the shift is exactly what corrupted the baked ISA
// references -- see GrowWithTrampolinesKeepsIsaReferenceConsistentWithSymbol.)
TEST(ElfView, GrowWithTrampolinesKeepsAllocSectionSymbols) {
  static constexpr uint64_t GrowthBytes = 8;

  comgr_test::KernelDescriptorElfOptions Opts;
  Opts.KernelName = "entry_kernel";
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(makeText(), Opts);

  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());

  Trampoline T;
  T.Bytes.assign(GrowthBytes, 0);
  std::vector<Trampoline> Trampolines;
  Trampolines.push_back(T);
  const uint8_t SNop[4] = {};
  std::unique_ptr<llvm::WritableMemoryBuffer> Out =
      ViewOrErr->growWithTrampolines(Trampolines, SNop);
  ASSERT_NE(Out, nullptr);

  uint8_t *OutData = reinterpret_cast<uint8_t *>(Out->getBufferStart());
  llvm::Expected<ElfView> OutView =
      ElfView::create(OutData, Out->getBufferSize());
  ASSERT_TRUE((bool)OutView) << llvm::toString(OutView.takeError());
  llvm::ArrayRef<KernelDescriptorInfo> KDs = OutView->kernelDescriptors();
  ASSERT_EQ(KDs.size(), 1u);
  EXPECT_EQ(KDs[0].VAddr, Obj.RodataAddr);
}

// gfx1250 "address of a global" idiom, as emitted for e.g. `x++` on a
// __managed__/__device__ global and observed in GPU_func of the
// Unit_hipModuleGetGlobal_Functional reproducer:
//
//   s_get_pc_i64 s[0:1]                         ; s[0:1] = addr of next insn
//   s_add_nc_u64 s[0:1], s[0:1], lit64(delta)   ; s[0:1] = that addr + delta
//
// The 64-bit literal is baked at link time (no relocation) and encodes the
// distance from the s_add instruction to the referenced symbol. Reading it back
// out of .text is exactly how the hardware resolves the global's address, so it
// is the ground truth any rewrite must keep consistent with the symbol table.
namespace {
constexpr uint8_t SGetPcI64SS01[4] = {0xBE, 0x80, 0x47, 0x00};
constexpr uint8_t SAddNcU64Lit[4] = {0xA9, 0x80, 0xFE, 0x00};
constexpr size_t GetPcOffset = 0;
constexpr size_t AddOpOffset = 4; // s_get_pc_i64 is one dword
constexpr size_t Lit64Offset = 8; // + s_add_nc_u64 opcode dword
constexpr size_t RefSeqSize = 16; // + 8-byte lit64

// Build a .text image containing the reference idiom. The literal is computed
// so that, loaded at TextAddr, the ISA resolves the reference to TargetVAddr.
std::vector<uint8_t> makeTextReferencing(uint64_t TextAddr,
                                         uint64_t TargetVAddr) {
  std::vector<uint8_t> Text(RefSeqSize, 0);
  std::memcpy(Text.data() + GetPcOffset, SGetPcI64SS01, sizeof(SGetPcI64SS01));
  std::memcpy(Text.data() + AddOpOffset, SAddNcU64Lit, sizeof(SAddNcU64Lit));
  // s_get_pc_i64 returns the address of the *following* instruction (the
  // s_add), so the PC base the add works from is TextAddr + AddOpOffset.
  const uint64_t PcBase = TextAddr + AddOpOffset;
  const uint64_t Lit = TargetVAddr - PcBase; // two's-complement; forward here
  std::memcpy(Text.data() + Lit64Offset, &Lit, sizeof(Lit));
  return Text;
}

// Decode the reference idiom out of a .text image loaded at TextAddr and return
// the virtual address the ISA resolves it to.
uint64_t decodeReferencedVAddr(const uint8_t *Text, uint64_t TextAddr) {
  uint64_t Lit = 0;
  std::memcpy(&Lit, Text + Lit64Offset, sizeof(Lit));
  return TextAddr + AddOpOffset + Lit;
}
} // namespace

// The real invariant: after appending trampolines, the address the *ISA*
// resolves a global reference to (decoded from the PC-relative literal in
// .text) must equal the address the *symbol table* reports for that global.
//
// This is the ELF-layer reproduction of Unit_hipModuleGetGlobal_Functional: the
// entry-trampoline rewrite grew .text, which shifted the referenced symbol by
// the trampoline size while leaving the baked literal pointing at the old
// location, so the ISA and the symbol table disagreed and the kernel
// dereferenced the wrong address. Here the descriptor symbol lives in .rodata
// (after .text), standing in for a global in a post-.text data section.
TEST(ElfView, GrowWithTrampolinesKeepsIsaReferenceConsistentWithSymbol) {
  // One 256-byte entry stub, matching the real
  // Unit_hipModuleGetGlobal_Functional reproducer's 0x100 shift.
  static constexpr uint64_t GrowthBytes = 0x100;

  comgr_test::KernelDescriptorElfOptions Opts;
  Opts.KernelName = "entry_kernel";
  // .text (at TextAddr) references the descriptor symbol in .rodata (at
  // RodataAddr), which sits after .text -- like a kernel referencing a global
  // in a post-.text data section.
  std::vector<uint8_t> Text =
      makeTextReferencing(Opts.TextAddr, Opts.RodataAddr);
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(Text, Opts);

  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());

  // Sanity: before the rewrite, the ISA reference and the symbol agree.
  ASSERT_EQ(decodeReferencedVAddr(ViewOrErr->textData(), ViewOrErr->textAddr()),
            Obj.RodataAddr);

  Trampoline T;
  T.Bytes.assign(GrowthBytes, 0);
  std::vector<Trampoline> Trampolines{T};
  const uint8_t SNop[4] = {};
  std::unique_ptr<llvm::WritableMemoryBuffer> Out =
      ViewOrErr->growWithTrampolines(Trampolines, SNop);
  ASSERT_NE(Out, nullptr);

  uint8_t *OutData = reinterpret_cast<uint8_t *>(Out->getBufferStart());
  llvm::Expected<ElfView> OutView =
      ElfView::create(OutData, Out->getBufferSize());
  ASSERT_TRUE((bool)OutView) << llvm::toString(OutView.takeError());

  // Decode whatever is now in .text into the address the ISA resolves the
  // reference to...
  const uint64_t IsaResolved =
      decodeReferencedVAddr(OutView->textData(), OutView->textAddr());
  // ...vs. the address the symbol table now reports for the same global.
  llvm::ArrayRef<KernelDescriptorInfo> KDs = OutView->kernelDescriptors();
  ASSERT_EQ(KDs.size(), 1u);
  const uint64_t SymbolVAddr = KDs[0].VAddr;

  EXPECT_EQ(IsaResolved, SymbolVAddr);
}

// A fully-linked code object's DWARF encodes *absolute* virtual addresses of
// code and data (DW_AT_low_pc, DW_OP_addr, the .debug_addr pool, .debug_line
// set_address) with no relocations. The old growWithTrampolines shifted
// post-.text virtual addresses and symbols but left .debug_* contents
// untouched, so any such address went stale by the trampoline size -- the
// debugger would resolve a global to the pre-shift location. This is the
// debug-info analogue of the ISA-literal corruption above; the no-shift model
// keeps both in agreement.
namespace {

// Minimal ET_DYN AMDGPU object:
//   [1] .text        (alloc, exec)  at TextAddr
//   [2] .data        (alloc, write) at DataAddr -- holds global "g"
//   [3] .debug_info  (non-alloc)    -- 8-byte absolute address DWARF would
//                                       encode for "g" (stands in for a
//                                       DW_AT_low_pc / DW_OP_addr / .debug_addr
//                                       entry)
//   [4] .symtab  [5] .strtab  [6] .shstrtab
// Both .data and .debug_info follow .text.
struct DwarfRefElf {
  std::vector<uint8_t> Bytes;
  uint64_t DataAddr = 0;
  unsigned DebugSectionIndex = 3;
  unsigned SymtabSectionIndex = 4;
  unsigned GlobalSymIndex = 1;
};

DwarfRefElf makeDwarfRefElf(uint64_t TextAddr = 0x1000,
                            uint64_t DataAddr = 0x2000) {
  using namespace llvm::ELF;
  constexpr uint64_t TextSize = 16;
  constexpr unsigned SymCount = 2; // null + "g"

  static const char ShStr[] =
      "\0.text\0.data\0.debug_info\0.symtab\0.strtab\0.shstrtab\0";
  constexpr uint32_t NameText = 1;
  constexpr uint32_t NameData = 7;
  constexpr uint32_t NameDebug = 13;
  constexpr uint32_t NameSymtab = 25;
  constexpr uint32_t NameStrtab = 33;
  constexpr uint32_t NameShstrtab = 41;

  static const char Str[] = "\0g\0";
  constexpr uint32_t GNameOff = 1;

  constexpr unsigned ShNum = 7;
  const uint64_t ShOff = sizeof(Elf64_Ehdr);
  const uint64_t TextOff =
      comgr_test::alignTo8(ShOff + ShNum * sizeof(Elf64_Shdr));
  const uint64_t DataOff = comgr_test::alignTo8(TextOff + TextSize);
  const uint64_t DebugOff = comgr_test::alignTo8(DataOff + 8);
  const uint64_t StrOff = comgr_test::alignTo8(DebugOff + 8);
  const uint64_t SymOff = comgr_test::alignTo8(StrOff + sizeof(Str));
  const uint64_t ShStrOff =
      comgr_test::alignTo8(SymOff + SymCount * sizeof(Elf64_Sym));
  const uint64_t BufSize = comgr_test::alignTo8(ShStrOff + sizeof(ShStr) + 64);

  DwarfRefElf R;
  R.Bytes.assign(BufSize, 0);
  R.DataAddr = DataAddr;
  uint8_t *B = R.Bytes.data();

  Elf64_Ehdr Ehdr = comgr_test::makeElf64Ehdr(EM_AMDGPU);
  Ehdr.e_ident[EI_OSABI] = ELFOSABI_AMDGPU_HSA;
  Ehdr.e_type = ET_DYN;
  Ehdr.e_version = EV_CURRENT;
  Ehdr.e_shoff = ShOff;
  Ehdr.e_ehsize = sizeof(Elf64_Ehdr);
  Ehdr.e_shentsize = sizeof(Elf64_Shdr);
  Ehdr.e_shnum = ShNum;
  Ehdr.e_shstrndx = 6;
  std::memcpy(B, &Ehdr, sizeof(Ehdr));

  std::memcpy(B + StrOff, Str, sizeof(Str));
  std::memcpy(B + ShStrOff, ShStr, sizeof(ShStr));

  // The absolute address DWARF encodes for "g".
  const uint64_t DebugAddr = DataAddr;
  std::memcpy(B + DebugOff, &DebugAddr, sizeof(DebugAddr));

  auto writeShdr = [&](unsigned Idx, const Elf64_Shdr &Sh) {
    std::memcpy(B + ShOff + Idx * sizeof(Elf64_Shdr), &Sh, sizeof(Sh));
  };

  Elf64_Shdr Text{};
  Text.sh_name = NameText;
  Text.sh_type = SHT_PROGBITS;
  Text.sh_flags = SHF_ALLOC | SHF_EXECINSTR;
  Text.sh_offset = TextOff;
  Text.sh_addr = TextAddr;
  Text.sh_size = TextSize;
  Text.sh_addralign = 4;
  writeShdr(1, Text);

  Elf64_Shdr Data{};
  Data.sh_name = NameData;
  Data.sh_type = SHT_PROGBITS;
  Data.sh_flags = SHF_ALLOC | SHF_WRITE;
  Data.sh_offset = DataOff;
  Data.sh_addr = DataAddr;
  Data.sh_size = 8;
  Data.sh_addralign = 8;
  writeShdr(2, Data);

  Elf64_Shdr Debug{};
  Debug.sh_name = NameDebug;
  Debug.sh_type = SHT_PROGBITS;
  Debug.sh_flags = 0; // non-alloc, like real .debug_* sections
  Debug.sh_offset = DebugOff;
  Debug.sh_size = 8;
  Debug.sh_addralign = 1;
  writeShdr(3, Debug);

  Elf64_Shdr Symtab{};
  Symtab.sh_name = NameSymtab;
  Symtab.sh_type = SHT_SYMTAB;
  Symtab.sh_offset = SymOff;
  Symtab.sh_size = SymCount * sizeof(Elf64_Sym);
  Symtab.sh_link = 5; // .strtab
  Symtab.sh_info = 1;
  Symtab.sh_entsize = sizeof(Elf64_Sym);
  writeShdr(4, Symtab);

  Elf64_Shdr Strtab{};
  Strtab.sh_name = NameStrtab;
  Strtab.sh_type = SHT_STRTAB;
  Strtab.sh_offset = StrOff;
  Strtab.sh_size = sizeof(Str);
  writeShdr(5, Strtab);

  Elf64_Shdr Shstr{};
  Shstr.sh_name = NameShstrtab;
  Shstr.sh_type = SHT_STRTAB;
  Shstr.sh_offset = ShStrOff;
  Shstr.sh_size = sizeof(ShStr);
  writeShdr(6, Shstr);

  Elf64_Sym G{};
  G.st_name = GNameOff;
  G.setBindingAndType(STB_GLOBAL, STT_OBJECT);
  G.st_shndx = 2; // .data
  G.st_value = DataAddr;
  G.st_size = 8;
  std::memcpy(B + SymOff + 1 * sizeof(Elf64_Sym), &G, sizeof(G));

  return R;
}

// Read an 8-byte value from section [Idx] at intra-section byte offset Off.
uint64_t readSectionU64(const ElfView &V, unsigned Idx, uint64_t Off) {
  uint64_t Val = 0;
  std::memcpy(
      &Val, V.data() + static_cast<uint64_t>(V.sections()[Idx].sh_offset) + Off,
      sizeof(Val));
  return Val;
}

// Read st_value of symbol SymIdx in the symbol table at section [SymtabIdx].
uint64_t readSymbolValue(const ElfView &V, unsigned SymtabIdx,
                         unsigned SymIdx) {
  llvm::ELF::Elf64_Sym Sym{};
  std::memcpy(&Sym,
              V.data() +
                  static_cast<uint64_t>(V.sections()[SymtabIdx].sh_offset) +
                  SymIdx * sizeof(llvm::ELF::Elf64_Sym),
              sizeof(Sym));
  return Sym.st_value;
}

} // namespace

// The invariant: after appending trampolines, the address the object's DWARF
// encodes for a global must still equal the address the symbol table reports
// for it. The no-shift model leaves both untouched, so they agree (the old
// shift moved the symbol but not the DWARF address). Debug-info analogue of
// GrowWithTrampolinesKeepsIsaReferenceConsistentWithSymbol.
TEST(ElfView, GrowWithTrampolinesKeepsDwarfConsistentWithSymbol) {
  static constexpr uint64_t GrowthBytes = 0x100;

  DwarfRefElf Obj = makeDwarfRefElf();

  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());

  // Sanity: before the rewrite, DWARF and the symbol agree.
  ASSERT_EQ(readSectionU64(*ViewOrErr, Obj.DebugSectionIndex, 0), Obj.DataAddr);
  ASSERT_EQ(
      readSymbolValue(*ViewOrErr, Obj.SymtabSectionIndex, Obj.GlobalSymIndex),
      Obj.DataAddr);

  Trampoline T;
  T.Bytes.assign(GrowthBytes, 0);
  std::vector<Trampoline> Trampolines{T};
  const uint8_t SNop[4] = {};
  std::unique_ptr<llvm::WritableMemoryBuffer> Out =
      ViewOrErr->growWithTrampolines(Trampolines, SNop);
  ASSERT_NE(Out, nullptr);

  uint8_t *OutData = reinterpret_cast<uint8_t *>(Out->getBufferStart());
  llvm::Expected<ElfView> OutView =
      ElfView::create(OutData, Out->getBufferSize());
  ASSERT_TRUE((bool)OutView) << llvm::toString(OutView.takeError());

  // Address DWARF still encodes for "g" ...
  const uint64_t DwarfAddr = readSectionU64(*OutView, Obj.DebugSectionIndex, 0);
  // ... vs. the address the symbol table now reports for "g".
  const uint64_t SymbolVAddr =
      readSymbolValue(*OutView, Obj.SymtabSectionIndex, Obj.GlobalSymIndex);

  EXPECT_EQ(DwarfAddr, SymbolVAddr);
}

// Covers: addKernelEntryTrampolineSymbols attaches a distinct, correctly
// placed `<kernel>.stub` symbol for every appended entry-trampoline stub, so a
// dispatch whose entry now points at a stub still resolves to a name.
//
// How: build a synthetic AMDGPU code object that has a .symtab, then grow .text
// by two entry-stub-sized (KernelEntryStubStride) blocks with
// growWithTrampolines -- mirroring the pass appending one stub per kernel.
// Call addKernelEntryTrampolineSymbols with two fixups that use distinct kernel
// names and the two stub offsets (0 and KernelEntryStubStride). Re-parse the
// returned buffer with llvm::object::ELFFile and, for each fixup, assert a
// "<name>.stub" symbol exists in .symtab that is (a) STT_FUNC, (b) defined in
// the .text section (st_shndx), (c) located at TextAddr + OldTextSize +
// StubTextOffset, and (d) sized to KernelEntryStubStride. Two fixups (rather
// than one) prove each stub gets its own name at its own address, not a single
// shared or mis-placed entry.
TEST(ElfView, AddKernelEntryTrampolineSymbolsNamesEachStub) {
  comgr_test::KernelDescriptorElfOptions Opts;
  Opts.KernelName = "entry_kernel";
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(makeText(), Opts);

  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());
  const unsigned TextIdx = ViewOrErr->textSectionIndex();
  const uint64_t TextAddr = ViewOrErr->textAddr();
  const uint64_t OldTextSize = ViewOrErr->textSize();

  // Grow .text by two entry-stub-sized blocks, mirroring the entry-trampoline
  // pass appending one stub per kernel.
  Trampoline Stub;
  Stub.Bytes.assign(2 * KernelEntryStubStride, 0xAA);
  std::vector<Trampoline> Growth{Stub};
  const uint8_t SNop[4] = {};
  std::unique_ptr<llvm::WritableMemoryBuffer> Grown =
      ViewOrErr->growWithTrampolines(Growth, SNop);
  ASSERT_NE(Grown, nullptr);

  // One fixup per appended stub; the names need not match real kernels, since
  // addKernelEntryTrampolineSymbols only attaches a symbol at each stub address.
  std::vector<KernelEntryTrampolineFixup> Fixups = {
      {"kernel_a", /*StubTextOffset=*/0, /*RequiredSgprs=*/10},
      {"kernel_b", /*StubTextOffset=*/KernelEntryStubStride, /*RequiredSgprs=*/12},
  };
  std::unique_ptr<llvm::WritableMemoryBuffer> WithSyms =
      addKernelEntryTrampolineSymbols(*Grown, TextIdx, TextAddr, OldTextSize,
                                      Fixups);
  ASSERT_NE(WithSyms, nullptr);

  using ELFT = llvm::object::ELF64LE;
  const uint8_t *Data =
      reinterpret_cast<const uint8_t *>(WithSyms->getBufferStart());
  llvm::Expected<llvm::object::ELFFile<ELFT>> FileOrErr =
      llvm::object::ELFFile<ELFT>::create(llvm::StringRef(
          reinterpret_cast<const char *>(Data), WithSyms->getBufferSize()));
  ASSERT_TRUE((bool)FileOrErr) << llvm::toString(FileOrErr.takeError());
  llvm::object::ELFFile<ELFT> &File = *FileOrErr;

  llvm::Expected<ELFT::ShdrRange> SecsOrErr = File.sections();
  ASSERT_TRUE((bool)SecsOrErr) << llvm::toString(SecsOrErr.takeError());
  const ELFT::Shdr *SymtabShdr = nullptr;
  for (const ELFT::Shdr &S : *SecsOrErr)
    if (S.sh_type == llvm::ELF::SHT_SYMTAB) {
      SymtabShdr = &S;
      break;
    }
  ASSERT_NE(SymtabShdr, nullptr);
  llvm::Expected<ELFT::SymRange> SymsOrErr = File.symbols(SymtabShdr);
  ASSERT_TRUE((bool)SymsOrErr) << llvm::toString(SymsOrErr.takeError());
  llvm::Expected<llvm::StringRef> StrTabOrErr =
      File.getStringTableForSymtab(*SymtabShdr);
  ASSERT_TRUE((bool)StrTabOrErr) << llvm::toString(StrTabOrErr.takeError());

  auto FindSym = [&](llvm::StringRef Name) -> const ELFT::Sym * {
    for (const ELFT::Sym &Sym : *SymsOrErr) {
      llvm::Expected<llvm::StringRef> N = Sym.getName(*StrTabOrErr);
      if (N && *N == Name)
        return &Sym;
    }
    return nullptr;
  };

  // Every appended stub must have a <kernel>.stub STT_FUNC symbol covering the
  // stub, in the .text section, at the stub's virtual address.
  for (const KernelEntryTrampolineFixup &F : Fixups) {
    const ELFT::Sym *Sym = FindSym(F.KernelName + ".stub");
    ASSERT_NE(Sym, nullptr) << "missing stub symbol for " << F.KernelName;
    EXPECT_EQ(static_cast<unsigned>(Sym->getType()),
              static_cast<unsigned>(llvm::ELF::STT_FUNC));
    EXPECT_EQ(Sym->st_shndx, TextIdx);
    EXPECT_EQ(Sym->st_value, TextAddr + OldTextSize + F.StubTextOffset);
    EXPECT_EQ(Sym->st_size, KernelEntryStubStride);
  }
}

TEST(ElfView, AddKernelEntryTrampolineSymbolsPreservesPhdr) {
  comgr_test::KernelDescriptorElfOptions Opts;
  Opts.KernelName = "entry_kernel";
  Opts.MetadataSgprCount = 8;
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(makeText(), Opts);

  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());
  const unsigned TextIdx = ViewOrErr->textSectionIndex();
  const uint64_t TextAddr = ViewOrErr->textAddr();
  const uint64_t OldTextSize = ViewOrErr->textSize();

  Trampoline Stub;
  Stub.Bytes.assign(2 * KernelEntryStubStride, 0xAA);
  std::vector<Trampoline> Growth{Stub};
  const uint8_t SNop[4] = {};
  std::unique_ptr<llvm::WritableMemoryBuffer> Grown =
      ViewOrErr->growWithTrampolines(Growth, SNop);
  ASSERT_NE(Grown, nullptr);

  std::vector<KernelEntryTrampolineFixup> Fixups = {
      {"kernel_a", /*StubTextOffset=*/0, /*RequiredSgprs=*/10},
      {"kernel_b", /*StubTextOffset=*/KernelEntryStubStride,
       /*RequiredSgprs=*/12},
  };
  std::unique_ptr<llvm::WritableMemoryBuffer> WithSyms =
      addKernelEntryTrampolineSymbols(*Grown, TextIdx, TextAddr, OldTextSize,
                                      Fixups);
  ASSERT_NE(WithSyms, nullptr);

  using ELFT = llvm::object::ELF64LE;
  llvm::Expected<llvm::object::ELFFile<ELFT>> FileOrErr =
      llvm::object::ELFFile<ELFT>::create(llvm::StringRef(
          WithSyms->getBufferStart(), WithSyms->getBufferSize()));
  ASSERT_TRUE((bool)FileOrErr) << llvm::toString(FileOrErr.takeError());

  llvm::Expected<ELFT::PhdrRange> PhdrsOrErr = FileOrErr->program_headers();
  ASSERT_TRUE((bool)PhdrsOrErr) << llvm::toString(PhdrsOrErr.takeError());
  EXPECT_GE(PhdrsOrErr->size(), 2u);

  bool FoundPoolLoad = false;
  for (const auto &Phdr : *PhdrsOrErr) {
    EXPECT_LE(Phdr.p_offset, WithSyms->getBufferSize());
    if (Phdr.p_type == llvm::ELF::PT_LOAD && (Phdr.p_flags & llvm::ELF::PF_X)) {
      FoundPoolLoad = true;
      EXPECT_GT(Phdr.p_filesz, 0u);
      ASSERT_LE(Phdr.p_offset + Phdr.p_filesz, WithSyms->getBufferSize());
      const uint8_t *PoolBytes =
          reinterpret_cast<const uint8_t *>(WithSyms->getBufferStart()) +
          Phdr.p_offset;
      for (uint64_t I = 0; I < Phdr.p_filesz; ++I)
        EXPECT_EQ(PoolBytes[I], 0xAA) << "pool content mismatch at byte " << I;
    }
  }
  EXPECT_TRUE(FoundPoolLoad) << "no executable PT_LOAD segment found";
}

TEST(ElfView, UpdateKernelDescriptorSgprCountUpdatesMetadataAndDescriptor) {
  comgr_test::KernelDescriptorElfOptions Opts;
  Opts.KernelName = "entry_kernel";
  Opts.MetadataSgprCount = 8;
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(makeText(), Opts);

  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());

  // Prime the metadata cache before the in-place update.
  EXPECT_EQ(ViewOrErr->getKernelSgprCount("entry_kernel"), 8u);
  ASSERT_TRUE(ViewOrErr->updateKernelDescriptorSgprCount("entry_kernel", 10));
  std::optional<unsigned> MetadataSgprs =
      ViewOrErr->getKernelSgprCount("entry_kernel");
  ASSERT_TRUE(MetadataSgprs.has_value());
  EXPECT_EQ(*MetadataSgprs, 10u);
  EXPECT_GE(readReservedSgprs(Obj.Bytes, Obj.KernelDescriptorOffset), 10u);
}

TEST(ElfView, UpdateGfx1250RevisionMetadataRetagsKernelInPlace) {
  comgr_test::KernelDescriptorElfOptions Opts;
  Opts.MetadataSgprCount = 8;
  Opts.MetadataGfx1250Revision = "B0";
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(makeText(), Opts);

  llvm::StringRef Before(reinterpret_cast<const char *>(Obj.Bytes.data()),
                         Obj.Bytes.size());
  EXPECT_NE(Before.find("B0"), llvm::StringRef::npos);

  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());
  ASSERT_TRUE(ViewOrErr->updateGfx1250RevisionMetadata("A0"));

  llvm::StringRef After(reinterpret_cast<const char *>(Obj.Bytes.data()),
                        Obj.Bytes.size());
  EXPECT_EQ(After.find("B0"), llvm::StringRef::npos);
  EXPECT_NE(After.find("A0"), llvm::StringRef::npos);
}

TEST(ElfView, UpdateKernelDescriptorSgprCountCanUpdateMetadataOnly) {
  comgr_test::KernelDescriptorElfOptions Opts;
  Opts.KernelName = "entry_kernel";
  Opts.MetadataSgprCount = 8;
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(makeText(), Opts);

  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());
  const unsigned DescriptorSgprsBefore =
      readReservedSgprs(Obj.Bytes, Obj.KernelDescriptorOffset);

  ASSERT_TRUE(ViewOrErr->updateKernelDescriptorSgprCount(
      "entry_kernel", 10, /*UpdateDescriptor=*/false));
  EXPECT_EQ(ViewOrErr->getKernelSgprCount("entry_kernel"), 10u);
  EXPECT_EQ(readReservedSgprs(Obj.Bytes, Obj.KernelDescriptorOffset),
            DescriptorSgprsBefore);
}

TEST(ElfView, UpdateKernelMetadataSgprCountsKeepsPrimedCacheCoherent) {
  comgr_test::KernelDescriptorElfOptions Opts;
  Opts.KernelName = "entry_kernel";
  Opts.MetadataSgprCount = 8;
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(makeText(), Opts);

  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());
  EXPECT_EQ(ViewOrErr->getKernelSgprCount("entry_kernel"), 8u);

  llvm::StringMap<unsigned> RequiredSgprs;
  RequiredSgprs.try_emplace("entry_kernel", 10u);
  ASSERT_TRUE(ViewOrErr->updateKernelMetadataSgprCounts(RequiredSgprs));
  EXPECT_EQ(ViewOrErr->getKernelSgprCount("entry_kernel"), 10u);
}

TEST(ElfView, UpdateKernelMetadataSgprCountsBatchesMixedRequirements) {
  comgr_test::KernelDescriptorElfOptions Opts;
  Opts.MetadataKernels = {{"needs_update", 8}, {"already_enough", 16}};
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(makeText(), Opts);

  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());

  llvm::StringMap<unsigned> RequiredSgprs;
  RequiredSgprs.try_emplace("needs_update", 10u);
  RequiredSgprs.try_emplace("already_enough", 12u);
  ASSERT_TRUE(ViewOrErr->updateKernelMetadataSgprCounts(RequiredSgprs));
  EXPECT_EQ(ViewOrErr->getKernelSgprCount("needs_update"), 10u);
  EXPECT_EQ(ViewOrErr->getKernelSgprCount("already_enough"), 16u);
}

TEST(ElfView, UpdateKernelMetadataSgprCountsRejectsAbsentKernelAtomically) {
  comgr_test::KernelDescriptorElfOptions Opts;
  Opts.MetadataKernels = {{"needs_update", 8}, {"already_enough", 16}};
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(makeText(), Opts);

  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());

  llvm::StringMap<unsigned> RequiredSgprs;
  RequiredSgprs.try_emplace("needs_update", 10u);
  RequiredSgprs.try_emplace("already_enough", 12u);
  RequiredSgprs.try_emplace("absent_kernel", 4u);
  EXPECT_FALSE(ViewOrErr->updateKernelMetadataSgprCounts(RequiredSgprs));
  EXPECT_EQ(ViewOrErr->getKernelSgprCount("needs_update"), 8u);
  EXPECT_EQ(ViewOrErr->getKernelSgprCount("already_enough"), 16u);
  EXPECT_EQ(ViewOrErr->getKernelSgprCount("absent_kernel"), std::nullopt);
}

TEST(ElfView, UpdateKernelDescriptorSgprCountMetadataOnlyRequiresMetadata) {
  comgr_test::KernelDescriptorElfOptions Opts;
  Opts.KernelName = "entry_kernel";
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(makeText(), Opts);

  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());
  const unsigned DescriptorSgprsBefore =
      readReservedSgprs(Obj.Bytes, Obj.KernelDescriptorOffset);

  EXPECT_FALSE(ViewOrErr->updateKernelDescriptorSgprCount(
      "entry_kernel", 10, /*UpdateDescriptor=*/false));
  EXPECT_EQ(readReservedSgprs(Obj.Bytes, Obj.KernelDescriptorOffset),
            DescriptorSgprsBefore);
}

TEST(ElfView, UpdateKernelDescriptorSgprCountRejectsMissingMetadataCount) {
  comgr_test::KernelDescriptorElfOptions Opts;
  Opts.KernelName = "entry_kernel";
  Opts.MetadataOmitSgprCount = true;
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(makeText(), Opts);

  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());

  EXPECT_FALSE(ViewOrErr->updateKernelDescriptorSgprCount("entry_kernel", 10));
  EXPECT_EQ(readReservedSgprs(Obj.Bytes, Obj.KernelDescriptorOffset), 8u);
}

TEST(ElfView, UpdateKernelDescriptorSgprCountRejectsMissingMetadataKernel) {
  comgr_test::KernelDescriptorElfOptions Opts;
  Opts.KernelName = "entry_kernel";
  Opts.MetadataKernelName = "other_kernel";
  Opts.MetadataSgprCount = 8;
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(makeText(), Opts);

  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());

  EXPECT_EQ(ViewOrErr->getKernelSgprCount("entry_kernel"), std::nullopt);
  EXPECT_FALSE(ViewOrErr->updateKernelDescriptorSgprCount("entry_kernel", 10));
  EXPECT_EQ(readReservedSgprs(Obj.Bytes, Obj.KernelDescriptorOffset), 8u);
}

TEST(ElfView, UpdateKernelDescriptorSgprCountRejectsNonIntegerMetadataCount) {
  comgr_test::KernelDescriptorElfOptions Opts;
  Opts.KernelName = "entry_kernel";
  Opts.MetadataSgprCountAsString = true;
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(makeText(), Opts);

  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());

  EXPECT_EQ(ViewOrErr->getKernelSgprCount("entry_kernel"), std::nullopt);
  EXPECT_FALSE(ViewOrErr->updateKernelDescriptorSgprCount("entry_kernel", 10));
  EXPECT_EQ(readReservedSgprs(Obj.Bytes, Obj.KernelDescriptorOffset), 8u);
}

TEST(ElfView, UpdateKernelDescriptorSgprCountRejectsMetadataSizeChange) {
  comgr_test::KernelDescriptorElfOptions Opts;
  Opts.KernelName = "entry_kernel";
  Opts.MetadataSgprCount = 9;
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(makeText(), Opts);

  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());

  EXPECT_FALSE(ViewOrErr->updateKernelDescriptorSgprCount("entry_kernel", 128));
  std::optional<unsigned> MetadataSgprs =
      ViewOrErr->getKernelSgprCount("entry_kernel");
  ASSERT_TRUE(MetadataSgprs.has_value());
  EXPECT_EQ(*MetadataSgprs, 9u);
  EXPECT_EQ(readReservedSgprs(Obj.Bytes, Obj.KernelDescriptorOffset), 8u);
}

TEST(ElfView, UpdateKernelDescriptorSgprCountRejectsDescriptorLimitFirst) {
  comgr_test::KernelDescriptorElfOptions Opts;
  Opts.KernelName = "entry_kernel";
  Opts.MetadataSgprCount = 200;
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(makeText(), Opts);

  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());

  EXPECT_FALSE(
      ViewOrErr->updateKernelDescriptorSgprCount("entry_kernel", 100000));
  std::optional<unsigned> MetadataSgprs =
      ViewOrErr->getKernelSgprCount("entry_kernel");
  ASSERT_TRUE(MetadataSgprs.has_value());
  EXPECT_EQ(*MetadataSgprs, 200u);
  EXPECT_EQ(readReservedSgprs(Obj.Bytes, Obj.KernelDescriptorOffset), 8u);
}
