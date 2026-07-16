//===- comgr-test-elf-utils.h - shared ELF builders for unit tests --------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef COMGR_TEST_UNIT_ELF_UTILS_H
#define COMGR_TEST_UNIT_ELF_UTILS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/BinaryFormat/MsgPackDocument.h"
#include "llvm/Support/AMDHSAKernelDescriptor.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace comgr_test {

// A zero-initialized 64-bit little-endian ELF header with valid magic and the
// given machine + e_flags. Callers set any further fields they need.
inline llvm::ELF::Elf64_Ehdr makeElf64Ehdr(uint16_t Machine,
                                           uint32_t Flags = 0) {
  using namespace llvm::ELF;
  Elf64_Ehdr Ehdr{};
  Ehdr.e_ident[EI_MAG0] = 0x7f;
  Ehdr.e_ident[EI_MAG1] = 'E';
  Ehdr.e_ident[EI_MAG2] = 'L';
  Ehdr.e_ident[EI_MAG3] = 'F';
  Ehdr.e_ident[EI_CLASS] = ELFCLASS64;
  Ehdr.e_ident[EI_DATA] = ELFDATA2LSB;
  Ehdr.e_ident[EI_VERSION] = EV_CURRENT;
  Ehdr.e_machine = Machine;
  Ehdr.e_flags = Flags;
  return Ehdr;
}

inline uint64_t alignTo4(uint64_t V) { return llvm::alignTo(V, 4); }

inline uint64_t alignTo8(uint64_t V) { return llvm::alignTo(V, 8); }

struct KernelDescriptorElfOptions {
  struct MetadataKernel {
    std::string Name;
    unsigned SgprCount = 0;
  };

  uint16_t ElfType = llvm::ELF::ET_DYN;
  std::string KernelName = "kernel";
  uint64_t TextAddr = 0x1000;
  uint64_t RodataAddr = 0x2000;
  bool EmitKernelDescriptorSymbol = true;
  // Emit the kernel entry STT_FUNC symbol with st_size == 0, matching AMDGPU
  // HSACO objects where the size lives on the .kd object symbol. Exercises the
  // nearest-preceding lookup in ElfView::findKernelAtAddress.
  bool ZeroSizeKernelSym = false;
  std::optional<uint64_t> KernelDescriptorSymbolValue;
  uint32_t GroupSegmentFixedSize = 0;
  uint32_t ComputePgmRsrc1 = 0;
  uint32_t ComputePgmRsrc3 = 0;
  std::optional<std::string> MetadataKernelName;
  std::optional<unsigned> MetadataSgprCount;
  std::optional<unsigned> MetadataVgprCount;
  std::optional<unsigned> MetadataMaxFlatWorkgroupSize;
  std::optional<unsigned> MetadataWavefrontSize;
  std::optional<std::string> MetadataGfx1250Revision;
  bool MetadataOmitSgprCount = false;
  bool MetadataSgprCountAsString = false;
  // When non-empty, emit these kernel metadata entries instead of the single
  // entry described by MetadataKernelName / MetadataSgprCount.
  std::vector<MetadataKernel> MetadataKernels;
  bool MetadataOmitVgprCount = false;
  bool MetadataVgprCountAsString = false;
};

struct KernelDescriptorElf {
  std::vector<uint8_t> Bytes;
  uint64_t RodataAddr = 0;
  uint64_t KernelDescriptorOffset = 0;
  int64_t EntryOffset = 0;
};

inline std::string
makeAmdgpuMetadataBlob(const KernelDescriptorElfOptions &Options) {
  llvm::msgpack::Document Doc;
  llvm::msgpack::MapDocNode Root = Doc.getRoot().getMap(/*Convert=*/true);
  llvm::msgpack::ArrayDocNode Kernels = Doc.getArrayNode();
  if (!Options.MetadataKernels.empty()) {
    for (const KernelDescriptorElfOptions::MetadataKernel &Spec :
         Options.MetadataKernels) {
      llvm::msgpack::MapDocNode Kernel = Doc.getMapNode();
      Kernel[".name"] = Doc.getNode(Spec.Name, /*Copy=*/true);
      Kernel[".sgpr_count"] = static_cast<uint64_t>(Spec.SgprCount);
      Kernels.push_back(Kernel);
    }
  } else {
    llvm::msgpack::MapDocNode Kernel = Doc.getMapNode();
    const std::string &MetadataKernelName = Options.MetadataKernelName
                                                ? *Options.MetadataKernelName
                                                : Options.KernelName;
    Kernel[".name"] = Doc.getNode(MetadataKernelName, /*Copy=*/true);
    if (!Options.MetadataOmitSgprCount) {
      if (Options.MetadataSgprCountAsString)
        Kernel[".sgpr_count"] = Doc.getNode("not-an-integer", /*Copy=*/true);
      else
        Kernel[".sgpr_count"] =
            static_cast<uint64_t>(Options.MetadataSgprCount.value_or(0));
    }
    if (!Options.MetadataOmitVgprCount) {
      if (Options.MetadataVgprCountAsString)
        Kernel[".vgpr_count"] = Doc.getNode("not-an-integer", /*Copy=*/true);
      else
        Kernel[".vgpr_count"] =
            static_cast<uint64_t>(Options.MetadataVgprCount.value_or(0));
    }
    if (Options.MetadataMaxFlatWorkgroupSize)
      Kernel[".max_flat_workgroup_size"] =
          static_cast<uint64_t>(*Options.MetadataMaxFlatWorkgroupSize);
    if (Options.MetadataWavefrontSize)
      Kernel[".wavefront_size"] =
          static_cast<uint64_t>(*Options.MetadataWavefrontSize);
    if (Options.MetadataGfx1250Revision)
      Kernel[".gfx1250_revision"] =
          Doc.getNode(*Options.MetadataGfx1250Revision, /*Copy=*/true);
    Kernels.push_back(Kernel);
  }
  Root["amdhsa.kernels"] = Kernels;

  std::string Blob;
  Doc.writeToBlob(Blob);
  return Blob;
}

inline void appendBytes(std::vector<uint8_t> &Out, const void *Data,
                        size_t Size) {
  const uint8_t *Begin = reinterpret_cast<const uint8_t *>(Data);
  Out.insert(Out.end(), Begin, Begin + Size);
}

inline void appendPadding(std::vector<uint8_t> &Out, uint64_t Alignment) {
  assert(llvm::isPowerOf2_64(Alignment));
  uint64_t PaddedSize =
      llvm::alignTo(static_cast<uint64_t>(Out.size()), Alignment);
  assert(PaddedSize <= std::numeric_limits<size_t>::max());
  Out.resize(static_cast<size_t>(PaddedSize), 0);
}

inline std::vector<uint8_t> makeAmdgpuMetadataNote(llvm::StringRef Blob) {
  using namespace llvm::ELF;

  static constexpr char NoteName[] = "AMDGPU";

  assert(Blob.size() <= std::numeric_limits<uint32_t>::max());

  Elf64_Nhdr Header{};
  Header.n_namesz = sizeof(NoteName);
  Header.n_descsz = static_cast<uint32_t>(Blob.size());
  Header.n_type = NT_AMDGPU_METADATA;

  std::vector<uint8_t> Note;
  appendBytes(Note, &Header, sizeof(Header));
  appendBytes(Note, NoteName, sizeof(NoteName));
  appendPadding(Note, 4);
  appendBytes(Note, Blob.data(), Blob.size());
  appendPadding(Note, 4);
  return Note;
}

inline KernelDescriptorElf
makeKernelDescriptorElf(llvm::ArrayRef<uint8_t> Text,
                        const KernelDescriptorElfOptions &Options = {}) {
  using namespace llvm::ELF;
  namespace hsa = llvm::amdhsa;

  static constexpr uint64_t ShOff = sizeof(Elf64_Ehdr);
  static constexpr uint64_t TextOffset = 0x240;
  static constexpr uint64_t KdBytes = sizeof(hsa::kernel_descriptor_t);
  static constexpr char ShStrTab[] =
      "\0.text\0.rodata\0.strtab\0.symtab\0.shstrtab\0";

  std::string StrTab;
  StrTab.push_back('\0');
  uint32_t KernelNameOff = StrTab.size();
  StrTab += Options.KernelName;
  StrTab.push_back('\0');
  uint32_t KdNameOff = StrTab.size();
  StrTab += Options.KernelName;
  StrTab += ".kd";
  StrTab.push_back('\0');

  const bool HasMetadataNote =
      Options.MetadataSgprCount || Options.MetadataGfx1250Revision ||
      Options.MetadataOmitSgprCount || Options.MetadataSgprCountAsString ||
      Options.MetadataMaxFlatWorkgroupSize || Options.MetadataWavefrontSize ||
      Options.MetadataVgprCount || Options.MetadataOmitVgprCount ||
      Options.MetadataVgprCountAsString || !Options.MetadataKernels.empty();
  std::vector<uint8_t> MetadataNote;
  if (HasMetadataNote) {
    std::string MetadataBlob = makeAmdgpuMetadataBlob(Options);
    MetadataNote = makeAmdgpuMetadataNote(MetadataBlob);
  }

  const uint64_t RodataOff = alignTo8(TextOffset + Text.size());
  const uint64_t StrTabOff = alignTo8(RodataOff + KdBytes);
  const uint64_t SymTabOff = alignTo8(StrTabOff + StrTab.size());
  const uint64_t SymCount = Options.EmitKernelDescriptorSymbol ? 3 : 2;
  const uint64_t ShStrTabOff =
      alignTo8(SymTabOff + SymCount * sizeof(Elf64_Sym));
  const uint64_t NoteOff =
      HasMetadataNote ? alignTo4(ShStrTabOff + sizeof(ShStrTab)) : 0;
  const uint64_t PhOff =
      HasMetadataNote ? alignTo8(NoteOff + MetadataNote.size()) : 0;
  const uint64_t ContentEnd = HasMetadataNote ? PhOff + sizeof(Elf64_Phdr)
                                              : ShStrTabOff + sizeof(ShStrTab);
  const uint64_t BufSize = alignTo8(ContentEnd + 64);

  KernelDescriptorElf Result;
  Result.Bytes.assign(BufSize, 0);
  Result.RodataAddr = Options.RodataAddr;
  Result.KernelDescriptorOffset = RodataOff;
  const uint64_t KernelDescriptorAddr =
      Options.KernelDescriptorSymbolValue.value_or(
          Options.ElfType == ET_REL ? 0 : Options.RodataAddr);
  Result.EntryOffset = static_cast<int64_t>(Options.TextAddr) -
                       static_cast<int64_t>(Options.RodataAddr);

  uint8_t *Buf = Result.Bytes.data();
  std::memcpy(Buf + TextOffset, Text.data(), Text.size());
  std::memcpy(Buf + StrTabOff, StrTab.data(), StrTab.size());
  std::memcpy(Buf + ShStrTabOff, ShStrTab, sizeof(ShStrTab));
  if (HasMetadataNote)
    std::memcpy(Buf + NoteOff, MetadataNote.data(), MetadataNote.size());

  Elf64_Ehdr Ehdr = makeElf64Ehdr(EM_AMDGPU);
  Ehdr.e_ident[EI_OSABI] = ELFOSABI_AMDGPU_HSA;
  Ehdr.e_type = Options.ElfType;
  Ehdr.e_version = EV_CURRENT;
  Ehdr.e_shoff = ShOff;
  if (HasMetadataNote) {
    Ehdr.e_phoff = PhOff;
    Ehdr.e_phentsize = sizeof(Elf64_Phdr);
    Ehdr.e_phnum = 1;
  }
  Ehdr.e_ehsize = sizeof(Elf64_Ehdr);
  Ehdr.e_shentsize = sizeof(Elf64_Shdr);
  Ehdr.e_shnum = 6;
  Ehdr.e_shstrndx = 5;
  std::memcpy(Buf, &Ehdr, sizeof(Ehdr));

  Elf64_Shdr TextSh{};
  TextSh.sh_name = 1;
  TextSh.sh_type = SHT_PROGBITS;
  TextSh.sh_flags = SHF_ALLOC | SHF_EXECINSTR;
  TextSh.sh_offset = TextOffset;
  TextSh.sh_addr = Options.TextAddr;
  TextSh.sh_size = Text.size();
  TextSh.sh_addralign = 4;
  std::memcpy(Buf + ShOff + 1 * sizeof(Elf64_Shdr), &TextSh, sizeof(TextSh));

  Elf64_Shdr RodataSh{};
  RodataSh.sh_name = 7;
  RodataSh.sh_type = SHT_PROGBITS;
  RodataSh.sh_flags = SHF_ALLOC;
  RodataSh.sh_offset = RodataOff;
  RodataSh.sh_addr = Options.RodataAddr;
  RodataSh.sh_size = KdBytes;
  RodataSh.sh_addralign = 8;
  std::memcpy(Buf + ShOff + 2 * sizeof(Elf64_Shdr), &RodataSh,
              sizeof(RodataSh));

  Elf64_Shdr StrtabSh{};
  StrtabSh.sh_name = 15;
  StrtabSh.sh_type = SHT_STRTAB;
  StrtabSh.sh_offset = StrTabOff;
  StrtabSh.sh_size = StrTab.size();
  std::memcpy(Buf + ShOff + 3 * sizeof(Elf64_Shdr), &StrtabSh,
              sizeof(StrtabSh));

  Elf64_Shdr SymtabSh{};
  SymtabSh.sh_name = 23;
  SymtabSh.sh_type = SHT_SYMTAB;
  SymtabSh.sh_offset = SymTabOff;
  SymtabSh.sh_size = SymCount * sizeof(Elf64_Sym);
  SymtabSh.sh_link = 3;
  SymtabSh.sh_entsize = sizeof(Elf64_Sym);
  std::memcpy(Buf + ShOff + 4 * sizeof(Elf64_Shdr), &SymtabSh,
              sizeof(SymtabSh));

  Elf64_Shdr ShstrSh{};
  ShstrSh.sh_name = 31;
  ShstrSh.sh_type = SHT_STRTAB;
  ShstrSh.sh_offset = ShStrTabOff;
  ShstrSh.sh_size = sizeof(ShStrTab);
  std::memcpy(Buf + ShOff + 5 * sizeof(Elf64_Shdr), &ShstrSh, sizeof(ShstrSh));

  if (HasMetadataNote) {
    Elf64_Phdr NotePhdr{};
    NotePhdr.p_type = PT_NOTE;
    NotePhdr.p_offset = NoteOff;
    NotePhdr.p_filesz = MetadataNote.size();
    NotePhdr.p_memsz = MetadataNote.size();
    NotePhdr.p_align = 4;
    std::memcpy(Buf + PhOff, &NotePhdr, sizeof(NotePhdr));
  }

  std::memcpy(
      Buf + RodataOff +
          offsetof(hsa::kernel_descriptor_t, kernel_code_entry_byte_offset),
      &Result.EntryOffset, sizeof(Result.EntryOffset));
  std::memcpy(Buf + RodataOff +
                  offsetof(hsa::kernel_descriptor_t, group_segment_fixed_size),
              &Options.GroupSegmentFixedSize,
              sizeof(Options.GroupSegmentFixedSize));
  std::memcpy(Buf + RodataOff +
                  offsetof(hsa::kernel_descriptor_t, compute_pgm_rsrc1),
              &Options.ComputePgmRsrc1, sizeof(Options.ComputePgmRsrc1));
  std::memcpy(Buf + RodataOff +
                  offsetof(hsa::kernel_descriptor_t, compute_pgm_rsrc3),
              &Options.ComputePgmRsrc3, sizeof(Options.ComputePgmRsrc3));

  Elf64_Sym KernelSym{};
  KernelSym.st_name = KernelNameOff;
  KernelSym.setBindingAndType(STB_GLOBAL, STT_FUNC);
  KernelSym.st_shndx = 1;
  KernelSym.st_value = Options.TextAddr;
  KernelSym.st_size = Options.ZeroSizeKernelSym ? 0 : Text.size();
  std::memcpy(Buf + SymTabOff + 1 * sizeof(Elf64_Sym), &KernelSym,
              sizeof(KernelSym));

  if (Options.EmitKernelDescriptorSymbol) {
    Elf64_Sym KdSym{};
    KdSym.st_name = KdNameOff;
    KdSym.setBindingAndType(STB_GLOBAL, STT_OBJECT);
    KdSym.st_shndx = 2;
    KdSym.st_value = KernelDescriptorAddr;
    KdSym.st_size = KdBytes;
    std::memcpy(Buf + SymTabOff + 2 * sizeof(Elf64_Sym), &KdSym, sizeof(KdSym));
  }

  return Result;
}

// Options for an ELF carrying several kernel descriptors, for tests that need
// more than one kernel (descriptor-cache dedup ordering, batched SGPR updates).
// Kept separate from makeKernelDescriptorElf so the single-kernel fixture used
// by most tests is untouched.
struct MultiKernelDescriptorElfOptions {
  struct Kernel {
    std::string Name;
    // STT_FUNC entry symbol value; must lie within [TextAddr,
    // TextAddr+TextSize).
    uint64_t EntryVAddr = 0;
    // STT_OBJECT `<name>.kd` symbol value; must be >= RodataAddr. The dedup
    // path keys on (name, KdVAddr), so repeat a (Name, KdVAddr) pair to emit a
    // duplicate symbol and vary KdVAddr to emit a distinct descriptor.
    uint64_t KdVAddr = 0;
    int64_t EntryOffset = 0; // kernel_code_entry_byte_offset in the descriptor
    uint32_t ComputePgmRsrc3 = 0;
    // When true, add an amdhsa.kernels metadata entry (with SgprCount) for this
    // kernel. Leave false to model a descriptor that has no metadata entry.
    bool EmitMetadata = false;
    unsigned MetadataSgprCount = 0;
  };

  uint16_t ElfType = llvm::ELF::ET_DYN;
  uint64_t TextAddr = 0x1000;
  uint64_t TextSize = 0x400;
  uint64_t RodataAddr = 0x2000;
  std::vector<Kernel> Kernels;
};

inline std::vector<uint8_t>
makeMultiKernelMetadataBlob(const MultiKernelDescriptorElfOptions &Options) {
  KernelDescriptorElfOptions MetaOpts;
  for (const MultiKernelDescriptorElfOptions::Kernel &K : Options.Kernels)
    if (K.EmitMetadata)
      MetaOpts.MetadataKernels.push_back({K.Name, K.MetadataSgprCount});
  if (MetaOpts.MetadataKernels.empty())
    return {};
  std::string Blob = makeAmdgpuMetadataBlob(MetaOpts);
  return makeAmdgpuMetadataNote(Blob);
}

inline std::vector<uint8_t>
makeMultiKernelDescriptorElf(const MultiKernelDescriptorElfOptions &Options) {
  using namespace llvm::ELF;
  namespace hsa = llvm::amdhsa;

  static constexpr uint64_t ShOff = sizeof(Elf64_Ehdr);
  static constexpr uint64_t TextOffset = 0x240;
  static constexpr uint64_t KdBytes = sizeof(hsa::kernel_descriptor_t);
  static constexpr char ShStrTab[] =
      "\0.text\0.rodata\0.strtab\0.symtab\0.shstrtab\0";

  const size_t NumKernels = Options.Kernels.size();
  assert(NumKernels > 0 && "multi-kernel ELF needs at least one kernel");

  // .rodata must span the highest descriptor block. Each KdVAddr must sit at or
  // above RodataAddr; distinct KdVAddrs get distinct blocks, duplicates reuse.
  uint64_t RodataSpan = KdBytes;
  for (const MultiKernelDescriptorElfOptions::Kernel &K : Options.Kernels) {
    assert(K.KdVAddr >= Options.RodataAddr && "kd vaddr below .rodata");
    assert(K.EntryVAddr >= Options.TextAddr &&
           K.EntryVAddr < Options.TextAddr + Options.TextSize &&
           "entry vaddr outside .text");
    RodataSpan = std::max(RodataSpan, K.KdVAddr - Options.RodataAddr + KdBytes);
  }

  // String table: null byte, then per kernel "<name>\0" and "<name>.kd\0".
  // Duplicates are emitted verbatim so repeated symbols stay independent.
  std::string StrTab;
  StrTab.push_back('\0');
  std::vector<uint32_t> FuncNameOff(NumKernels), KdNameOff(NumKernels);
  for (size_t I = 0; I < NumKernels; ++I) {
    FuncNameOff[I] = static_cast<uint32_t>(StrTab.size());
    StrTab += Options.Kernels[I].Name;
    StrTab.push_back('\0');
    KdNameOff[I] = static_cast<uint32_t>(StrTab.size());
    StrTab += Options.Kernels[I].Name;
    StrTab += ".kd";
    StrTab.push_back('\0');
  }

  std::vector<uint8_t> MetadataNote = makeMultiKernelMetadataBlob(Options);
  const bool HasMetadataNote = !MetadataNote.empty();

  // Symbols: null [0], then per kernel a STT_FUNC entry and STT_OBJECT .kd.
  const uint64_t SymCount = 1 + 2 * NumKernels;

  const uint64_t RodataOff = alignTo8(TextOffset + Options.TextSize);
  const uint64_t StrTabOff = alignTo8(RodataOff + RodataSpan);
  const uint64_t SymTabOff = alignTo8(StrTabOff + StrTab.size());
  const uint64_t ShStrTabOff =
      alignTo8(SymTabOff + SymCount * sizeof(Elf64_Sym));
  const uint64_t NoteOff =
      HasMetadataNote ? alignTo4(ShStrTabOff + sizeof(ShStrTab)) : 0;
  const uint64_t PhOff =
      HasMetadataNote ? alignTo8(NoteOff + MetadataNote.size()) : 0;
  const uint64_t ContentEnd = HasMetadataNote ? PhOff + sizeof(Elf64_Phdr)
                                              : ShStrTabOff + sizeof(ShStrTab);
  const uint64_t BufSize = alignTo8(ContentEnd + 64);

  std::vector<uint8_t> Bytes(BufSize, 0);
  uint8_t *Buf = Bytes.data();
  std::memcpy(Buf + StrTabOff, StrTab.data(), StrTab.size());
  std::memcpy(Buf + ShStrTabOff, ShStrTab, sizeof(ShStrTab));
  if (HasMetadataNote)
    std::memcpy(Buf + NoteOff, MetadataNote.data(), MetadataNote.size());

  Elf64_Ehdr Ehdr = makeElf64Ehdr(EM_AMDGPU);
  Ehdr.e_ident[EI_OSABI] = ELFOSABI_AMDGPU_HSA;
  Ehdr.e_type = Options.ElfType;
  Ehdr.e_version = EV_CURRENT;
  Ehdr.e_shoff = ShOff;
  if (HasMetadataNote) {
    Ehdr.e_phoff = PhOff;
    Ehdr.e_phentsize = sizeof(Elf64_Phdr);
    Ehdr.e_phnum = 1;
  }
  Ehdr.e_ehsize = sizeof(Elf64_Ehdr);
  Ehdr.e_shentsize = sizeof(Elf64_Shdr);
  Ehdr.e_shnum = 6;
  Ehdr.e_shstrndx = 5;
  std::memcpy(Buf, &Ehdr, sizeof(Ehdr));

  Elf64_Shdr TextSh{};
  TextSh.sh_name = 1;
  TextSh.sh_type = SHT_PROGBITS;
  TextSh.sh_flags = SHF_ALLOC | SHF_EXECINSTR;
  TextSh.sh_offset = TextOffset;
  TextSh.sh_addr = Options.TextAddr;
  TextSh.sh_size = Options.TextSize;
  TextSh.sh_addralign = 4;
  std::memcpy(Buf + ShOff + 1 * sizeof(Elf64_Shdr), &TextSh, sizeof(TextSh));

  Elf64_Shdr RodataSh{};
  RodataSh.sh_name = 7;
  RodataSh.sh_type = SHT_PROGBITS;
  RodataSh.sh_flags = SHF_ALLOC;
  RodataSh.sh_offset = RodataOff;
  RodataSh.sh_addr = Options.RodataAddr;
  RodataSh.sh_size = RodataSpan;
  RodataSh.sh_addralign = 8;
  std::memcpy(Buf + ShOff + 2 * sizeof(Elf64_Shdr), &RodataSh,
              sizeof(RodataSh));

  Elf64_Shdr StrtabSh{};
  StrtabSh.sh_name = 15;
  StrtabSh.sh_type = SHT_STRTAB;
  StrtabSh.sh_offset = StrTabOff;
  StrtabSh.sh_size = StrTab.size();
  std::memcpy(Buf + ShOff + 3 * sizeof(Elf64_Shdr), &StrtabSh,
              sizeof(StrtabSh));

  Elf64_Shdr SymtabSh{};
  SymtabSh.sh_name = 23;
  SymtabSh.sh_type = SHT_SYMTAB;
  SymtabSh.sh_offset = SymTabOff;
  SymtabSh.sh_size = SymCount * sizeof(Elf64_Sym);
  SymtabSh.sh_link = 3; // .strtab section index
  SymtabSh.sh_info = 1; // one past the last local symbol (only sym[0] is local)
  SymtabSh.sh_entsize = sizeof(Elf64_Sym);
  std::memcpy(Buf + ShOff + 4 * sizeof(Elf64_Shdr), &SymtabSh,
              sizeof(SymtabSh));

  Elf64_Shdr ShstrSh{};
  ShstrSh.sh_name = 31;
  ShstrSh.sh_type = SHT_STRTAB;
  ShstrSh.sh_offset = ShStrTabOff;
  ShstrSh.sh_size = sizeof(ShStrTab);
  std::memcpy(Buf + ShOff + 5 * sizeof(Elf64_Shdr), &ShstrSh, sizeof(ShstrSh));

  if (HasMetadataNote) {
    Elf64_Phdr NotePhdr{};
    NotePhdr.p_type = PT_NOTE;
    NotePhdr.p_offset = NoteOff;
    NotePhdr.p_filesz = MetadataNote.size();
    NotePhdr.p_memsz = MetadataNote.size();
    NotePhdr.p_align = 4;
    std::memcpy(Buf + PhOff, &NotePhdr, sizeof(NotePhdr));
  }

  // Descriptor blocks and symbols, one pair per kernel entry.
  for (size_t I = 0; I < NumKernels; ++I) {
    const MultiKernelDescriptorElfOptions::Kernel &K = Options.Kernels[I];
    const uint64_t KdBlockOff = RodataOff + (K.KdVAddr - Options.RodataAddr);
    std::memcpy(
        Buf + KdBlockOff +
            offsetof(hsa::kernel_descriptor_t, kernel_code_entry_byte_offset),
        &K.EntryOffset, sizeof(K.EntryOffset));
    std::memcpy(Buf + KdBlockOff +
                    offsetof(hsa::kernel_descriptor_t, compute_pgm_rsrc3),
                &K.ComputePgmRsrc3, sizeof(K.ComputePgmRsrc3));

    Elf64_Sym FuncSym{};
    FuncSym.st_name = FuncNameOff[I];
    FuncSym.setBindingAndType(STB_GLOBAL, STT_FUNC);
    FuncSym.st_shndx = 1;
    FuncSym.st_value = K.EntryVAddr;
    FuncSym.st_size = 0x40;
    std::memcpy(Buf + SymTabOff + (1 + 2 * I) * sizeof(Elf64_Sym), &FuncSym,
                sizeof(FuncSym));

    Elf64_Sym KdSym{};
    KdSym.st_name = KdNameOff[I];
    KdSym.setBindingAndType(STB_GLOBAL, STT_OBJECT);
    KdSym.st_shndx = 2;
    KdSym.st_value = K.KdVAddr;
    KdSym.st_size = KdBytes;
    std::memcpy(Buf + SymTabOff + (2 + 2 * I) * sizeof(Elf64_Sym), &KdSym,
                sizeof(KdSym));
  }

  return Bytes;
}

} // namespace comgr_test

#endif // COMGR_TEST_UNIT_ELF_UTILS_H
