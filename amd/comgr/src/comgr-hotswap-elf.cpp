//===- comgr-hotswap-elf.cpp - ELF helpers and trampoline growth ----------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of hotswap::ElfView and the free-function ELF helpers.
/// Parses are delegated to llvm::object::ELFFile; there is no hand-rolled
/// section/symbol cache.
///
//===----------------------------------------------------------------------===//

#include "comgr-hotswap-internal.h"

#include "llvm/ADT/StringExtras.h"

using namespace llvm;

namespace COMGR {
namespace hotswap {

using Ehdr = ELF::Elf64_Ehdr;
using Shdr = ELF::Elf64_Shdr;
using Phdr = ELF::Elf64_Phdr;
using ELFT = ElfView::ELFT;
using ELFFileT = ElfView::ELFFileT;

// -- applyByteReplace ---------------------------------------------------------

bool applyByteReplace(const RewriteRule &Rule, uint64_t InstOffset,
                      uint32_t InstSize, uint8_t *Text, uint64_t TextSize,
                      const LLVMState &S) {
  if (InstOffset + InstSize > TextSize)
    return false;
  const size_t ReplaceSize = Rule.ReplaceBytes.size();
  if (ReplaceSize > InstSize)
    return false;
  if (S.SNopBytes.size() != MinInstSize)
    return false;
  std::memcpy(Text + InstOffset, Rule.ReplaceBytes.data(), ReplaceSize);
  uint64_t PadOffset = InstOffset + ReplaceSize;
  uint64_t Remaining = InstSize - ReplaceSize;
  while (Remaining >= MinInstSize) {
    std::memcpy(Text + PadOffset, S.SNopBytes.data(), MinInstSize);
    PadOffset += MinInstSize;
    Remaining -= MinInstSize;
  }
  return true;
}

// -- findNearestSled ----------------------------------------------------------

NopSled *findNearestSled(std::vector<NopSled> &Sleds, uint64_t Offset,
                         uint64_t Needed) {
  NopSled *Best = nullptr;
  int64_t BestDist = INT64_MAX;
  for (NopSled &Sled : Sleds) {
    if (Sled.WritePos + Needed > Sled.End)
      continue;
    int64_t Dist = std::abs(static_cast<int64_t>(Sled.WritePos) -
                            static_cast<int64_t>(Offset));
    if (Dist < MaxSledDistance && Dist < BestDist) {
      Best = &Sled;
      BestDist = Dist;
    }
  }
  return Best;
}

// -- ElfView::create ----------------------------------------------------------

Expected<ElfView> ElfView::create(uint8_t *Data, size_t Size) {
  // Data/Size are kept as factory parameters to document that the caller
  // must hand in a mutable buffer (hotswap mutates bytes through the
  // resulting ElfView). Once ELFFile is constructed, it owns the structural
  // view over these same bytes and we do not need to store Data/Size
  // separately -- ELFFile::base() / ELFFile::getBufSize() alias them.
  Expected<ELFFileT> FileOrErr =
      ELFFileT::create(StringRef(reinterpret_cast<const char *>(Data), Size));
  if (!FileOrErr)
    return FileOrErr.takeError();

  const ELFFileT &File = *FileOrErr;
  Expected<ELFT::ShdrRange> SectionsOrErr = File.sections();
  if (!SectionsOrErr)
    return SectionsOrErr.takeError();
  ELFT::ShdrRange Sections = *SectionsOrErr;

  const ELFT::Shdr *Text = nullptr;
  unsigned TextIdx = 0;
  unsigned Idx = 0;
  for (const ELFT::Shdr &Shdr : Sections) {
    Expected<StringRef> NameOrErr = File.getSectionName(Shdr);
    if (!NameOrErr) {
      consumeError(NameOrErr.takeError());
      ++Idx;
      continue;
    }
    if (*NameOrErr == ".text" && Shdr.sh_offset + Shdr.sh_size <= Size) {
      Text = &Shdr;
      TextIdx = Idx;
      break;
    }
    ++Idx;
  }
  if (!Text)
    return createStringError(object::object_error::parse_failed,
                             "no .text section found");
  return ElfView(std::move(*FileOrErr), Sections, Text, TextIdx);
}

// -- ElfView::findKernelAtOffset ----------------------------------------------

std::string ElfView::findKernelAtOffset(uint64_t TextOffset) const {
  for (const ELFT::Shdr &SymShdr : Sections) {
    if (SymShdr.sh_type != ELF::SHT_SYMTAB &&
        SymShdr.sh_type != ELF::SHT_DYNSYM)
      continue;

    Expected<ELFT::SymRange> SymsOrErr = File.symbols(&SymShdr);
    if (!SymsOrErr) {
      consumeError(SymsOrErr.takeError());
      continue;
    }
    Expected<StringRef> StrTabOrErr =
        File.getStringTableForSymtab(SymShdr, Sections);
    if (!StrTabOrErr) {
      consumeError(StrTabOrErr.takeError());
      continue;
    }

    for (const ELFT::Sym &Sym : *SymsOrErr) {
      if (Sym.getType() != ELF::STT_FUNC && Sym.getType() != ELF::STT_GNU_IFUNC)
        continue;
      if (Sym.st_shndx != TextSectionIndex)
        continue;
      if (TextOffset < Sym.st_value || TextOffset >= Sym.st_value + Sym.st_size)
        continue;
      Expected<StringRef> NameOrErr = Sym.getName(*StrTabOrErr);
      if (!NameOrErr) {
        log() << "hotswap: error: findKernelAtOffset: function symbol "
              << "covering offset 0x" << utohexstr(TextOffset)
              << " has unreadable name: " << toString(NameOrErr.takeError())
              << "\n";
        return "";
      }
      return NameOrErr->str();
    }
  }
  log() << "hotswap: findKernelAtOffset: no function symbol covers offset 0x"
        << utohexstr(TextOffset) << " in .text.\n";
  return "";
}

// -- ElfView::findKernelDescriptor --------------------------------------------

uint8_t *ElfView::findKernelDescriptor(StringRef KernelName) {
  std::string KdName = (KernelName + ".kd").str();
  for (const ELFT::Shdr &SymShdr : Sections) {
    if (SymShdr.sh_type != ELF::SHT_SYMTAB &&
        SymShdr.sh_type != ELF::SHT_DYNSYM)
      continue;

    Expected<ELFT::SymRange> SymsOrErr = File.symbols(&SymShdr);
    if (!SymsOrErr) {
      consumeError(SymsOrErr.takeError());
      continue;
    }
    Expected<StringRef> StrTabOrErr =
        File.getStringTableForSymtab(SymShdr, Sections);
    if (!StrTabOrErr) {
      consumeError(StrTabOrErr.takeError());
      continue;
    }

    for (const ELFT::Sym &Sym : *SymsOrErr) {
      Expected<StringRef> NameOrErr = Sym.getName(*StrTabOrErr);
      if (!NameOrErr) {
        consumeError(NameOrErr.takeError());
        continue;
      }
      if (*NameOrErr != KdName)
        continue;
      unsigned Shndx = Sym.st_shndx;
      Expected<const ELFT::Shdr *> HostShdrOrErr = File.getSection(Shndx);
      if (!HostShdrOrErr) {
        consumeError(HostShdrOrErr.takeError());
        continue;
      }
      const ELFT::Shdr &HostShdr = **HostShdrOrErr;
      if (Sym.st_value < HostShdr.sh_addr)
        continue;
      uint64_t FileOffset =
          HostShdr.sh_offset + (Sym.st_value - HostShdr.sh_addr);
      if (FileOffset + KdSize > size())
        continue;
      return data() + FileOffset;
    }
  }
  return nullptr;
}

// -- ElfView::getKernelVgprCount ----------------------------------------------

std::optional<unsigned>
ElfView::getKernelVgprCount(StringRef KernelName,
                            unsigned VgprGranuleSize) const {
  if (VgprGranuleSize == 0) {
    log() << "hotswap: error: getKernelVgprCount: VgprGranuleSize is 0 for "
          << "kernel '" << KernelName << "'.\n";
    return std::nullopt;
  }
  namespace hsa = amdhsa;
  // findKernelDescriptor never writes through the returned pointer in this
  // call path but is shared (non-const) with updateKernelDescriptor. The
  // const_cast on `this` keeps the read-only accessor const-correct without
  // duplicating the lookup helper.
  uint8_t *Kd = const_cast<ElfView *>(this)->findKernelDescriptor(KernelName);
  if (!Kd) {
    log() << "hotswap: error: getKernelVgprCount: kernel descriptor symbol '"
          << KernelName << ".kd' not found.\n";
    return std::nullopt;
  }
  uint32_t Rsrc1;
  std::memcpy(&Rsrc1,
              Kd + offsetof(hsa::kernel_descriptor_t, compute_pgm_rsrc1),
              sizeof(Rsrc1));
  uint32_t Granulated = AMDHSA_BITS_GET(
      Rsrc1, hsa::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT);
  return (Granulated + 1) * VgprGranuleSize;
}

// Reads the static (compile-time-fixed) LDS allocation from the kernel
// descriptor's group_segment_fixed_size field. Dynamic LDS is added by the
// host at dispatch time and is not visible here -- see the declaration's
// doc comment for the full lower-bound caveat.

std::optional<uint32_t>
ElfView::getKernelStaticLdsSize(StringRef KernelName) const {
  namespace hsa = amdhsa;
  // findKernelDescriptor never writes through the returned pointer in this
  // call path but is shared (non-const) with updateKernelDescriptor. The
  // const_cast on `this` keeps the read-only accessor const-correct without
  // duplicating the lookup helper.
  const uint8_t *Kd =
      const_cast<ElfView *>(this)->findKernelDescriptor(KernelName);
  if (!Kd) {
    log() << "hotswap: error: getKernelStaticLdsSize: kernel descriptor "
          << "symbol '" << KernelName << ".kd' not found.\n";
    return std::nullopt;
  }
  uint32_t LdsSize;
  std::memcpy(&LdsSize,
              Kd + offsetof(hsa::kernel_descriptor_t, group_segment_fixed_size),
              sizeof(LdsSize));
  return LdsSize;
}

// -- ElfView::updateKernelDescriptor ------------------------------------------

void ElfView::updateKernelDescriptor(StringRef KernelName, unsigned ExtraVgprs,
                                     unsigned ExtraSgprs,
                                     unsigned VgprGranuleSize,
                                     unsigned SgprGranuleSize) {
  namespace hsa = amdhsa;
  uint8_t *Kd = findKernelDescriptor(KernelName);
  if (!Kd) {
    log() << "hotswap: error: updateKernelDescriptor: kernel descriptor "
          << "symbol '" << KernelName << ".kd' not found; requested "
          << "+" << ExtraVgprs << " VGPRs / +" << ExtraSgprs
          << " SGPRs silently dropped.\n";
    return;
  }

  uint32_t Rsrc1;
  std::memcpy(&Rsrc1,
              Kd + offsetof(hsa::kernel_descriptor_t, compute_pgm_rsrc1),
              sizeof(Rsrc1));
  if (ExtraVgprs != 0 && VgprGranuleSize != 0) {
    uint32_t Current = AMDHSA_BITS_GET(
        Rsrc1, hsa::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT);
    uint32_t MaxGran = static_cast<uint32_t>(
        hsa::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT >>
        hsa::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT_SHIFT);
    unsigned Extra = (ExtraVgprs + VgprGranuleSize - 1) / VgprGranuleSize;
    AMDHSA_BITS_SET(Rsrc1,
                    hsa::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT,
                    std::min<uint32_t>(Current + Extra, MaxGran));
  }
  if (ExtraSgprs != 0 && SgprGranuleSize != 0) {
    uint32_t Current = AMDHSA_BITS_GET(
        Rsrc1, hsa::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT);
    uint32_t MaxGran = static_cast<uint32_t>(
        hsa::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT >>
        hsa::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT_SHIFT);
    unsigned Extra = (ExtraSgprs + SgprGranuleSize - 1) / SgprGranuleSize;
    AMDHSA_BITS_SET(Rsrc1,
                    hsa::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                    std::min<uint32_t>(Current + Extra, MaxGran));
  }
  std::memcpy(Kd + offsetof(hsa::kernel_descriptor_t, compute_pgm_rsrc1),
              &Rsrc1, sizeof(Rsrc1));
}

// -- Section/program header adjustment for trampoline growth ------------------

static void adjustSectionHeaders(uint8_t *Elf, size_t ElfSize,
                                 uint64_t TextOffset, uint64_t TextSize,
                                 size_t TrampTotal) {
  if (ElfSize < sizeof(Ehdr))
    return;

  uint64_t TextEnd = TextOffset + TextSize;
  uint64_t Shoff;
  uint16_t Shentsize;
  uint16_t Shnum;
  std::memcpy(&Shoff, Elf + offsetof(Ehdr, e_shoff), sizeof(Shoff));
  std::memcpy(&Shentsize, Elf + offsetof(Ehdr, e_shentsize), sizeof(Shentsize));
  std::memcpy(&Shnum, Elf + offsetof(Ehdr, e_shnum), sizeof(Shnum));
  if (Shentsize < sizeof(Shdr))
    return;

  if (Shoff >= TextEnd) {
    uint64_t NewShoff = Shoff + TrampTotal;
    std::memcpy(Elf + offsetof(Ehdr, e_shoff), &NewShoff, sizeof(NewShoff));
    Shoff = NewShoff;
  }

  for (uint16_t I = 0; I < Shnum; ++I) {
    uint64_t ShPos = Shoff + static_cast<uint64_t>(I) * Shentsize;
    if (ShPos + sizeof(Shdr) > ElfSize)
      break;
    uint8_t *Sh = Elf + ShPos;
    uint64_t ShOffset;
    std::memcpy(&ShOffset, Sh + offsetof(Shdr, sh_offset), sizeof(ShOffset));

    if (ShOffset == TextOffset) {
      uint64_t NewTextSize = TextSize + TrampTotal;
      std::memcpy(Sh + offsetof(Shdr, sh_size), &NewTextSize,
                  sizeof(NewTextSize));
    } else if (ShOffset > TextOffset) {
      uint64_t NewOffset = ShOffset + TrampTotal;
      std::memcpy(Sh + offsetof(Shdr, sh_offset), &NewOffset,
                  sizeof(NewOffset));
      uint64_t ShFlags;
      std::memcpy(&ShFlags, Sh + offsetof(Shdr, sh_flags), sizeof(ShFlags));
      if (ShFlags & ELF::SHF_ALLOC) {
        uint64_t ShAddr;
        std::memcpy(&ShAddr, Sh + offsetof(Shdr, sh_addr), sizeof(ShAddr));
        ShAddr += TrampTotal;
        std::memcpy(Sh + offsetof(Shdr, sh_addr), &ShAddr, sizeof(ShAddr));
      }
    }
  }
}

static void adjustProgramHeaders(uint8_t *Elf, size_t ElfSize,
                                 uint64_t TextOffset, uint64_t TextSize,
                                 size_t TrampTotal) {
  if (ElfSize < sizeof(Ehdr))
    return;

  uint64_t TextEnd = TextOffset + TextSize;
  uint64_t Phoff;
  uint16_t Phentsize;
  uint16_t Phnum;
  std::memcpy(&Phoff, Elf + offsetof(Ehdr, e_phoff), sizeof(Phoff));
  std::memcpy(&Phentsize, Elf + offsetof(Ehdr, e_phentsize), sizeof(Phentsize));
  std::memcpy(&Phnum, Elf + offsetof(Ehdr, e_phnum), sizeof(Phnum));
  if (Phentsize < sizeof(Phdr))
    return;

  for (uint16_t I = 0; I < Phnum; ++I) {
    uint64_t PhPos = Phoff + static_cast<uint64_t>(I) * Phentsize;
    if (PhPos + sizeof(Phdr) > ElfSize)
      break;
    uint8_t *Ph = Elf + PhPos;
    uint64_t POffset;
    uint64_t PFilesz;
    uint64_t PMemsz;
    std::memcpy(&POffset, Ph + offsetof(Phdr, p_offset), sizeof(POffset));
    std::memcpy(&PFilesz, Ph + offsetof(Phdr, p_filesz), sizeof(PFilesz));
    std::memcpy(&PMemsz, Ph + offsetof(Phdr, p_memsz), sizeof(PMemsz));

    if (POffset <= TextOffset && POffset + PFilesz >= TextEnd) {
      PFilesz += TrampTotal;
      PMemsz += TrampTotal;
      std::memcpy(Ph + offsetof(Phdr, p_filesz), &PFilesz, sizeof(PFilesz));
      std::memcpy(Ph + offsetof(Phdr, p_memsz), &PMemsz, sizeof(PMemsz));
    } else if (POffset > TextOffset) {
      POffset += TrampTotal;
      std::memcpy(Ph + offsetof(Phdr, p_offset), &POffset, sizeof(POffset));
      uint64_t PVaddr;
      std::memcpy(&PVaddr, Ph + offsetof(Phdr, p_vaddr), sizeof(PVaddr));
      PVaddr += TrampTotal;
      std::memcpy(Ph + offsetof(Phdr, p_vaddr), &PVaddr, sizeof(PVaddr));
      uint64_t PPaddr;
      std::memcpy(&PPaddr, Ph + offsetof(Phdr, p_paddr), sizeof(PPaddr));
      PPaddr += TrampTotal;
      std::memcpy(Ph + offsetof(Phdr, p_paddr), &PPaddr, sizeof(PPaddr));
    }
  }
}

// -- ElfView::growWithTrampolines ---------------------------------------------

std::unique_ptr<WritableMemoryBuffer>
ElfView::growWithTrampolines(ArrayRef<Trampoline> Trampolines,
                             ArrayRef<uint8_t> SNopBytes) const {
  const size_t InputSize = size();
  const uint8_t *Input = data();

  size_t TrampTotal = 0;
  for (const Trampoline &T : Trampolines)
    TrampTotal += T.Bytes.size();
  if (TrampTotal == 0) {
    log() << "hotswap: growWithTrampolines: no trampolines to insert; "
          << "returning empty result.\n";
    return nullptr;
  }
  if (TrampTotal > SIZE_MAX - InputSize) {
    log() << "hotswap: error: growWithTrampolines: trampoline bytes ("
          << TrampTotal << ") + existing ELF size (" << InputSize
          << ") overflow size_t.\n";
    return nullptr;
  }

  uint64_t TextEnd = textOffset() + textSize();

  // Pad TrampTotal to the maximum alignment of all post-.text sections so
  // that shifting file offsets preserves sh_addralign invariants. The
  // sh_addr update in adjustSectionHeaders is still gated on SHF_ALLOC.
  uint64_t MaxPostTextAlign = 1;
  for (const ELFT::Shdr &Shdr : Sections) {
    if (Shdr.sh_offset <= textOffset())
      continue;
    if (Shdr.sh_addralign > MaxPostTextAlign)
      MaxPostTextAlign = Shdr.sh_addralign;
  }
  size_t PaddedTrampTotal = llvm::alignTo(TrampTotal, MaxPostTextAlign);
  if (PaddedTrampTotal > SIZE_MAX - InputSize) {
    log() << "hotswap: error: growWithTrampolines: padded trampoline bytes ("
          << PaddedTrampTotal << ") + ELF size (" << InputSize
          << ") overflow size_t.\n";
    return nullptr;
  }
  size_t PadBytes = PaddedTrampTotal - TrampTotal;

  const size_t NewSize = InputSize + PaddedTrampTotal;
  std::unique_ptr<WritableMemoryBuffer> Buf =
      WritableMemoryBuffer::getNewUninitMemBuffer(NewSize);
  if (!Buf) {
    log() << "hotswap: error: growWithTrampolines: "
          << "WritableMemoryBuffer::getNewUninitMemBuffer(" << NewSize
          << ") failed (out of memory).\n";
    return nullptr;
  }

  uint8_t *Out = reinterpret_cast<uint8_t *>(Buf->getBufferStart());
  std::memcpy(Out, Input, TextEnd);
  uint64_t Pos = TextEnd;
  for (const Trampoline &T : Trampolines) {
    std::memcpy(Out + Pos, T.Bytes.data(), T.Bytes.size());
    Pos += T.Bytes.size();
  }
  if (PadBytes > 0 && SNopBytes.size() == MinInstSize) {
    for (size_t I = 0; I < PadBytes; I += MinInstSize)
      std::memcpy(Out + Pos + I, SNopBytes.data(), MinInstSize);
    Pos += PadBytes;
  } else if (PadBytes > 0) {
    std::memset(Out + Pos, 0, PadBytes);
    Pos += PadBytes;
  }
  if (TextEnd < InputSize)
    std::memcpy(Out + Pos, Input + TextEnd, InputSize - TextEnd);

  adjustSectionHeaders(Out, NewSize, textOffset(), textSize(),
                       PaddedTrampTotal);
  adjustProgramHeaders(Out, NewSize, textOffset(), textSize(),
                       PaddedTrampTotal);
  log() << "hotswap: growWithTrampolines: grew ELF from " << InputSize << " to "
        << NewSize << " bytes (" << Trampolines.size() << " trampoline"
        << (Trampolines.size() == 1 ? "" : "s") << ", " << TrampTotal
        << " trampoline bytes + " << PadBytes << " alignment padding).\n";
  return Buf;
}

} // namespace hotswap
} // namespace COMGR
