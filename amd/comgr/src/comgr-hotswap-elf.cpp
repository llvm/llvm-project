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
#include "llvm/ADT/Twine.h"
#include "llvm/BinaryFormat/MsgPackDocument.h"
#include "llvm/Support/CheckedArithmetic.h"

#include <algorithm>
#include <limits>

using namespace llvm;

namespace COMGR {
namespace hotswap {

using Ehdr = ELF::Elf64_Ehdr;
using Shdr = ELF::Elf64_Shdr;
using Phdr = ELF::Elf64_Phdr;
using ELFT = ElfView::ELFT;
using ELFFileT = ElfView::ELFFileT;

static constexpr unsigned SgprEncodingGranule = 8;

enum class MetadataSgprUpdateStatus {
  NotFound,
  Found,
  Error,
};

static std::optional<uint64_t> checkedAdd(uint64_t LHS, uint64_t RHS,
                                          StringRef Context) {
  std::optional<uint64_t> Result = checkedAddUnsigned(LHS, RHS);
  if (Result)
    return Result;

  log() << "hotswap: error: " << Context << " overflows uint64_t.\n";
  return std::nullopt;
}

static std::optional<size_t>
checkedAlignToSize(size_t Value, uint64_t Alignment, StringRef Context) {
  if (Alignment <= 1)
    return Value;
  uint64_t Value64 = static_cast<uint64_t>(Value);
  uint64_t Remainder = Value64 % Alignment;
  if (Remainder == 0)
    return Value;
  std::optional<uint64_t> Aligned =
      checkedAdd(Value64, Alignment - Remainder, Context);
  if (!Aligned)
    return std::nullopt;
  if (*Aligned > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    log() << "hotswap: error: " << Context << " exceeds size_t.\n";
    return std::nullopt;
  }
  return static_cast<size_t>(*Aligned);
}

static std::optional<uint64_t> checkedSectionFileOffset(const ELFT::Shdr &Sec,
                                                        uint64_t VAddr,
                                                        uint64_t AccessSize,
                                                        uint64_t FileSize,
                                                        StringRef Context) {
  if (VAddr < Sec.sh_addr) {
    log() << "hotswap: error: " << Context << " has vaddr 0x"
          << utohexstr(VAddr) << " before containing section vaddr 0x"
          << utohexstr(Sec.sh_addr) << ".\n";
    return std::nullopt;
  }

  uint64_t Delta = VAddr - Sec.sh_addr;
  std::optional<uint64_t> FileOffset =
      checkedAdd(Sec.sh_offset, Delta, (Twine(Context) + " file offset").str());
  if (!FileOffset)
    return std::nullopt;

  if (AccessSize > FileSize || *FileOffset > FileSize - AccessSize) {
    log() << "hotswap: error: " << Context
          << " extends past end of ELF at file offset 0x"
          << utohexstr(*FileOffset) << ".\n";
    return std::nullopt;
  }
  return FileOffset;
}

static std::optional<unsigned>
readSgprCountMetadataNode(const msgpack::DocNode &SgprNode,
                          StringRef KernelName, StringRef Context) {
  if (SgprNode.getKind() == msgpack::Type::UInt) {
    uint64_t SgprCount = SgprNode.getUInt();
    if (SgprCount > std::numeric_limits<unsigned>::max()) {
      log() << "hotswap: error: " << Context << ": .sgpr_count for '"
            << KernelName << "' exceeds unsigned.\n";
      return std::nullopt;
    }
    return static_cast<unsigned>(SgprCount);
  }

  if (SgprNode.getKind() == msgpack::Type::Int) {
    int64_t SgprCount = SgprNode.getInt();
    if (SgprCount < 0 || static_cast<uint64_t>(SgprCount) >
                             std::numeric_limits<unsigned>::max()) {
      log() << "hotswap: error: " << Context << ": .sgpr_count for '"
            << KernelName << "' is outside unsigned range.\n";
      return std::nullopt;
    }
    return static_cast<unsigned>(SgprCount);
  }

  log() << "hotswap: error: " << Context << ": .sgpr_count for '" << KernelName
        << "' is not an integer.\n";
  return std::nullopt;
}

static MetadataSgprUpdateStatus
updateKernelMetadataSgprCount(uint8_t *Elf, const ELFFileT &File,
                              StringRef KernelName, unsigned RequiredSgprs) {
  Expected<ELFT::PhdrRange> PhdrsOrErr = File.program_headers();
  if (!PhdrsOrErr) {
    log() << "hotswap: error: updateKernelMetadataSgprCount: failed to read "
          << "program headers: " << toString(PhdrsOrErr.takeError()) << "\n";
    return MetadataSgprUpdateStatus::Error;
  }

  bool SawMetadataNote = false;
  for (const ELFT::Phdr &Phdr : *PhdrsOrErr) {
    if (Phdr.p_type != ELF::PT_NOTE)
      continue;

    Error Err = Error::success();
    for (ELFT::Note Note : File.notes(Phdr, Err)) {
      if (Note.getName() != "AMDGPU" ||
          Note.getType() != ELF::NT_AMDGPU_METADATA)
        continue;
      SawMetadataNote = true;

      ArrayRef<uint8_t> Desc = Note.getDesc(4);
      if (Desc.empty()) {
        log() << "hotswap: error: updateKernelMetadataSgprCount: AMDGPU "
              << "metadata note has an empty descriptor.\n";
        return MetadataSgprUpdateStatus::Error;
      }

      StringRef Blob(reinterpret_cast<const char *>(Desc.data()), Desc.size());
      msgpack::Document Doc;
      if (!Doc.readFromBlob(Blob, false)) {
        log() << "hotswap: error: updateKernelMetadataSgprCount: failed to "
              << "parse AMDGPU metadata note.\n";
        return MetadataSgprUpdateStatus::Error;
      }

      msgpack::DocNode Root = Doc.getRoot();
      if (!Root.isMap()) {
        log() << "hotswap: error: updateKernelMetadataSgprCount: AMDGPU "
              << "metadata root is not a map.\n";
        return MetadataSgprUpdateStatus::Error;
      }

      msgpack::MapDocNode &RootMap = Root.getMap();
      msgpack::DocNode::MapTy::iterator KernelsIt =
          RootMap.find("amdhsa.kernels");
      if (KernelsIt == RootMap.end() || !KernelsIt->second.isArray())
        continue;

      msgpack::ArrayDocNode &KernelArray = KernelsIt->second.getArray();
      for (msgpack::DocNode &KNode : KernelArray) {
        if (!KNode.isMap())
          continue;

        msgpack::MapDocNode &KMap = KNode.getMap();
        msgpack::DocNode::MapTy::iterator NameIt = KMap.find(".name");
        if (NameIt == KMap.end() || !NameIt->second.isString() ||
            NameIt->second.getString() != KernelName)
          continue;

        msgpack::DocNode::MapTy::iterator SgprIt = KMap.find(".sgpr_count");
        if (SgprIt == KMap.end()) {
          log() << "hotswap: error: updateKernelMetadataSgprCount: metadata "
                << "for kernel '" << KernelName << "' has no .sgpr_count.\n";
          return MetadataSgprUpdateStatus::Error;
        }

        std::optional<unsigned> CurrentSgprs = readSgprCountMetadataNode(
            SgprIt->second, KernelName, "updateKernelMetadataSgprCount");
        if (!CurrentSgprs)
          return MetadataSgprUpdateStatus::Error;
        if (RequiredSgprs <= *CurrentSgprs)
          return MetadataSgprUpdateStatus::Found;

        SgprIt->second = static_cast<uint64_t>(RequiredSgprs);
        std::string NewBlob;
        Doc.writeToBlob(NewBlob);
        if (NewBlob.size() != Blob.size()) {
          log() << "hotswap: error: updateKernelMetadataSgprCount: updating "
                << ".sgpr_count for '" << KernelName << "' changes metadata "
                << "note size from " << Blob.size() << " to " << NewBlob.size()
                << " bytes; in-place rewrite cannot preserve ELF layout.\n";
          return MetadataSgprUpdateStatus::Error;
        }

        const uint8_t *DescBegin = Desc.data();
        if (DescBegin < File.base() || DescBegin >= File.end()) {
          log() << "hotswap: error: updateKernelMetadataSgprCount: metadata "
                << "descriptor pointer is outside the ELF buffer.\n";
          return MetadataSgprUpdateStatus::Error;
        }
        size_t DescOffset = DescBegin - File.base();
        if (Desc.size() > File.getBufSize() ||
            DescOffset > File.getBufSize() - Desc.size()) {
          log() << "hotswap: error: updateKernelMetadataSgprCount: metadata "
                << "descriptor extends past the ELF buffer.\n";
          return MetadataSgprUpdateStatus::Error;
        }

        std::memcpy(Elf + DescOffset, NewBlob.data(), NewBlob.size());
        return MetadataSgprUpdateStatus::Found;
      }
    }

    if (Err) {
      log() << "hotswap: error: updateKernelMetadataSgprCount: failed to "
            << "iterate AMDGPU notes: " << toString(std::move(Err)) << "\n";
      return MetadataSgprUpdateStatus::Error;
    }
  }

  if (SawMetadataNote) {
    log() << "hotswap: error: updateKernelMetadataSgprCount: AMDGPU metadata "
          << "has no entry for kernel '" << KernelName << "'.\n";
    return MetadataSgprUpdateStatus::Error;
  }
  return MetadataSgprUpdateStatus::NotFound;
}

// -- applyByteReplace ---------------------------------------------------------

bool applyByteReplace(const RewriteRule &Rule, uint64_t InstOffset,
                      uint32_t InstSize, uint8_t *Text, uint64_t TextSize,
                      const LLVMState &S) {
  if (InstOffset > TextSize || InstSize > TextSize - InstOffset) {
    log() << "hotswap: error: applyByteReplace: instruction range [0x"
          << utohexstr(InstOffset) << ", 0x"
          << utohexstr(InstOffset + static_cast<uint64_t>(InstSize))
          << ") extends past .text size 0x" << utohexstr(TextSize) << ".\n";
    return false;
  }
  const size_t ReplaceSize = Rule.ReplaceBytes.size();
  if (ReplaceSize > InstSize) {
    log() << "hotswap: error: applyByteReplace: replacement size "
          << ReplaceSize << " exceeds original instruction size " << InstSize
          << " at .text offset 0x" << utohexstr(InstOffset) << ".\n";
    return false;
  }
  if (S.SNopBytes.size() != MinInstSize) {
    log() << "hotswap: error: applyByteReplace: cached s_nop size "
          << S.SNopBytes.size() << " does not match expected size "
          << MinInstSize << ".\n";
    return false;
  }
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
  // Select the nearest-preceding STT_FUNC symbol (greatest st_value <=
  // TextOffset). AMDGPU kernel entry symbols often have st_size == 0, so an
  // exact containment test never matches; honor st_size only when non-zero.
  //
  // The nearest-preceding function could be a non-kernel device function; the
  // guard below rejects that case so callers never scratch-allocate against a
  // non-kernel context.
  bool Found = false;
  uint64_t BestValue = 0;
  std::string BestName;

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
      if (TextOffset < Sym.st_value)
        continue;
      // Honor an explicit upper bound; skip it for zero-size symbols.
      if (Sym.st_size != 0 && TextOffset >= Sym.st_value + Sym.st_size)
        continue;
      if (Found && Sym.st_value <= BestValue)
        continue;
      Expected<StringRef> NameOrErr = Sym.getName(*StrTabOrErr);
      if (!NameOrErr) {
        consumeError(NameOrErr.takeError());
        continue;
      }
      Found = true;
      BestValue = Sym.st_value;
      BestName = NameOrErr->str();
    }
  }

  if (Found) {
    // Confirm the selected symbol is actually a kernel: every kernel carries a
    // "<name>.kd" descriptor symbol, whereas a plain device function does not.
    // This is the same descriptor lookup getKernelVgprCount performs, so a real
    // kernel is never rejected; a non-kernel is reported as "not found" so the
    // caller declines instead of scratch-allocating against a wrong context.
    if (const_cast<ElfView *>(this)->findKernelDescriptor(BestName)) {
      return BestName;
    }
    log() << "hotswap: findKernelAtOffset: nearest function symbol '"
          << BestName << "' preceding offset 0x" << utohexstr(TextOffset)
          << " has no .kd descriptor (not a kernel); treating as no match.\n";
    return "";
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
      std::optional<uint64_t> FileOffset = checkedSectionFileOffset(
          HostShdr, Sym.st_value, KdSize, size(),
          (Twine("findKernelDescriptor: descriptor symbol '") + *NameOrErr +
           "'")
              .str());
      if (!FileOffset)
        continue;
      return data() + *FileOffset;
    }
  }
  return nullptr;
}

// -- ElfView::kernelDescriptors -----------------------------------------------

std::vector<KernelDescriptorInfo> ElfView::kernelDescriptors() const {
  namespace hsa = amdhsa;
  std::vector<KernelDescriptorInfo> Result;

  for (const ELFT::Shdr &SymShdr : Sections) {
    if (SymShdr.sh_type != ELF::SHT_SYMTAB &&
        SymShdr.sh_type != ELF::SHT_DYNSYM)
      continue;

    Expected<ELFT::SymRange> SymsOrErr = File.symbols(&SymShdr);
    if (!SymsOrErr) {
      log() << "hotswap: error: kernelDescriptors: failed to read symbols: "
            << toString(SymsOrErr.takeError()) << "\n";
      continue;
    }
    Expected<StringRef> StrTabOrErr =
        File.getStringTableForSymtab(SymShdr, Sections);
    if (!StrTabOrErr) {
      log() << "hotswap: error: kernelDescriptors: failed to read symbol "
            << "string table: " << toString(StrTabOrErr.takeError()) << "\n";
      continue;
    }

    for (const ELFT::Sym &Sym : *SymsOrErr) {
      Expected<StringRef> NameOrErr = Sym.getName(*StrTabOrErr);
      if (!NameOrErr) {
        log() << "hotswap: error: kernelDescriptors: failed to read symbol "
              << "name: " << toString(NameOrErr.takeError()) << "\n";
        continue;
      }
      if (!NameOrErr->ends_with(".kd"))
        continue;

      Expected<const ELFT::Shdr *> HostShdrOrErr =
          File.getSection(Sym.st_shndx);
      if (!HostShdrOrErr) {
        log() << "hotswap: error: kernelDescriptors: descriptor symbol '"
              << *NameOrErr << "' has unreadable section index " << Sym.st_shndx
              << ": " << toString(HostShdrOrErr.takeError()) << "\n";
        continue;
      }
      const ELFT::Shdr &HostShdr = **HostShdrOrErr;
      std::optional<uint64_t> FileOffset = checkedSectionFileOffset(
          HostShdr, Sym.st_value, KdSize, size(),
          (Twine("kernelDescriptors: descriptor symbol '") + *NameOrErr + "'")
              .str());
      if (!FileOffset)
        continue;

      int64_t EntryOffset = 0;
      std::memcpy(
          &EntryOffset,
          data() + *FileOffset +
              offsetof(hsa::kernel_descriptor_t, kernel_code_entry_byte_offset),
          sizeof(EntryOffset));

      std::string KernelName = NameOrErr->drop_back(3).str();
      const bool Seen = std::any_of(
          Result.begin(), Result.end(), [&](const KernelDescriptorInfo &Info) {
            return Info.KernelName == KernelName && Info.VAddr == Sym.st_value;
          });
      if (!Seen)
        Result.push_back({std::move(KernelName), Sym.st_value, EntryOffset});
    }
  }

  return Result;
}

std::optional<uint64_t>
ElfView::getKernelDescriptorVAddr(StringRef KernelName) const {
  for (const KernelDescriptorInfo &Info : kernelDescriptors()) {
    if (Info.KernelName == KernelName)
      return Info.VAddr;
  }
  return std::nullopt;
}

bool ElfView::updateKernelDescriptorEntryOffset(StringRef KernelName,
                                                int64_t NewEntryOffset) {
  namespace hsa = amdhsa;
  uint8_t *Kd = findKernelDescriptor(KernelName);
  if (!Kd) {
    log() << "hotswap: error: updateKernelDescriptorEntryOffset: kernel "
          << "descriptor symbol '" << KernelName << ".kd' not found.\n";
    return false;
  }
  std::memcpy(
      Kd + offsetof(hsa::kernel_descriptor_t, kernel_code_entry_byte_offset),
      &NewEntryOffset, sizeof(NewEntryOffset));
  return true;
}

bool ElfView::updateKernelDescriptorSgprCount(StringRef KernelName,
                                              unsigned RequiredSgprs) {
  namespace hsa = amdhsa;
  if (RequiredSgprs == 0)
    return true;

  uint8_t *Kd = findKernelDescriptor(KernelName);
  if (!Kd) {
    log() << "hotswap: error: updateKernelDescriptorSgprCount: kernel "
          << "descriptor symbol '" << KernelName << ".kd' not found.\n";
    return false;
  }

  uint32_t Rsrc1 = 0;
  std::memcpy(&Rsrc1,
              Kd + offsetof(hsa::kernel_descriptor_t, compute_pgm_rsrc1),
              sizeof(Rsrc1));

  uint32_t CurrentGranulated = AMDHSA_BITS_GET(
      Rsrc1, hsa::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT);
  uint64_t CurrentSgprs =
      (static_cast<uint64_t>(CurrentGranulated) + 1) * SgprEncodingGranule;

  std::optional<uint32_t> RequiredGranulated;
  if (RequiredSgprs > CurrentSgprs) {
    uint64_t RequiredGranulated64 =
        (static_cast<uint64_t>(RequiredSgprs) + SgprEncodingGranule - 1) /
            SgprEncodingGranule -
        1;
    uint32_t MaxGranulated = static_cast<uint32_t>(
        hsa::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT >>
        hsa::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT_SHIFT);
    if (RequiredGranulated64 > MaxGranulated) {
      log() << "hotswap: error: updateKernelDescriptorSgprCount: kernel '"
            << KernelName << "' needs " << RequiredSgprs
            << " SGPRs, which exceeds the descriptor encoding limit.\n";
      return false;
    }
    RequiredGranulated = static_cast<uint32_t>(RequiredGranulated64);
  }

  MetadataSgprUpdateStatus MetadataStatus =
      updateKernelMetadataSgprCount(data(), File, KernelName, RequiredSgprs);
  if (MetadataStatus == MetadataSgprUpdateStatus::Error)
    return false;
  // NotFound is allowed for minimal code objects without AMDGPU metadata; in
  // that case the descriptor field remains the only SGPR count to update.

  if (!RequiredGranulated)
    return true;

  AMDHSA_BITS_SET(Rsrc1, hsa::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                  *RequiredGranulated);
  std::memcpy(Kd + offsetof(hsa::kernel_descriptor_t, compute_pgm_rsrc1),
              &Rsrc1, sizeof(Rsrc1));
  return true;
}

std::optional<uint32_t>
ElfView::getKernelDescriptorInstPrefSize(StringRef KernelName,
                                         StringRef TargetCpu) const {
  namespace hsa = amdhsa;
  uint8_t *Kd = const_cast<ElfView *>(this)->findKernelDescriptor(KernelName);
  if (!Kd) {
    log() << "hotswap: error: getKernelDescriptorInstPrefSize: kernel "
          << "descriptor symbol '" << KernelName << ".kd' not found.\n";
    return std::nullopt;
  }

  uint32_t Rsrc3 = 0;
  std::memcpy(&Rsrc3,
              Kd + offsetof(hsa::kernel_descriptor_t, compute_pgm_rsrc3),
              sizeof(Rsrc3));

  if (TargetCpu.starts_with("gfx12")) {
    return AMDHSA_BITS_GET(Rsrc3,
                           hsa::COMPUTE_PGM_RSRC3_GFX12_PLUS_INST_PREF_SIZE);
  }

  log() << "hotswap: error: getKernelDescriptorInstPrefSize: unsupported "
        << "target CPU '" << TargetCpu << "' for kernel '" << KernelName
        << "'.\n";
  return std::nullopt;
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
  uint64_t VgprCount =
      (static_cast<uint64_t>(Granulated) + 1) * VgprGranuleSize;
  if (VgprCount > std::numeric_limits<unsigned>::max()) {
    log() << "hotswap: error: getKernelVgprCount: descriptor VGPR count for '"
          << KernelName << "' exceeds unsigned.\n";
    return std::nullopt;
  }
  return static_cast<unsigned>(VgprCount);
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

// -- ElfView::getKernelSgprCount ----------------------------------------------
//
// Reads .sgpr_count from the amdhsa.kernels msgpack metadata note.
// On GFX10+ GRANULATED_WAVEFRONT_SGPR_COUNT in the kernel descriptor is
// architecturally reserved (must be zero), so the metadata note is the
// preferred source. Falls back to the KD field when no metadata note is
// present (e.g. minimal test ELFs assembled with -nostdlib).

std::optional<unsigned>
ElfView::getKernelSgprCount(StringRef KernelName) const {
  // --- Try msgpack metadata note first. ---
  Expected<ELFT::PhdrRange> PhdrsOrErr = File.program_headers();
  bool SawMetadataNote = false;
  if (PhdrsOrErr) {
    for (const ELFT::Phdr &Phdr : *PhdrsOrErr) {
      if (Phdr.p_type != ELF::PT_NOTE)
        continue;
      Error Err = Error::success();
      for (ELFT::Note Note : File.notes(Phdr, Err)) {
        if (Note.getName() != "AMDGPU" ||
            Note.getType() != ELF::NT_AMDGPU_METADATA)
          continue;
        SawMetadataNote = true;

        ArrayRef<uint8_t> Desc = Note.getDesc(4);
        if (Desc.empty()) {
          log() << "hotswap: error: getKernelSgprCount: AMDGPU metadata note "
                << "has an empty descriptor.\n";
          return std::nullopt;
        }

        StringRef Blob(reinterpret_cast<const char *>(Desc.data()),
                       Desc.size());
        msgpack::Document Doc;
        if (!Doc.readFromBlob(Blob, false)) {
          log() << "hotswap: error: getKernelSgprCount: failed to parse "
                << "AMDGPU metadata note.\n";
          return std::nullopt;
        }

        msgpack::DocNode Root = Doc.getRoot();
        if (!Root.isMap()) {
          log() << "hotswap: error: getKernelSgprCount: AMDGPU metadata root "
                << "is not a map.\n";
          return std::nullopt;
        }
        msgpack::MapDocNode &RootMap = Root.getMap();
        msgpack::DocNode::MapTy::iterator KernelsIt =
            RootMap.find("amdhsa.kernels");
        if (KernelsIt == RootMap.end() || !KernelsIt->second.isArray())
          continue;

        msgpack::ArrayDocNode &KernelArray = KernelsIt->second.getArray();
        for (msgpack::DocNode &KNode : KernelArray) {
          if (!KNode.isMap())
            continue;
          msgpack::MapDocNode &KMap = KNode.getMap();
          msgpack::DocNode::MapTy::iterator NameIt = KMap.find(".name");
          if (NameIt == KMap.end() || !NameIt->second.isString() ||
              NameIt->second.getString() != KernelName)
            continue;

          msgpack::DocNode::MapTy::iterator SgprIt = KMap.find(".sgpr_count");
          if (SgprIt == KMap.end()) {
            log() << "hotswap: error: getKernelSgprCount: metadata for kernel '"
                  << KernelName << "' has no .sgpr_count.\n";
            return std::nullopt;
          }
          return readSgprCountMetadataNode(SgprIt->second, KernelName,
                                           "getKernelSgprCount");
        }
      }
      if (Err) {
        log() << "hotswap: error: getKernelSgprCount: failed to iterate "
              << "AMDGPU notes: " << toString(std::move(Err)) << "\n";
        return std::nullopt;
      }
    }
  } else {
    log() << "hotswap: error: getKernelSgprCount: failed to read program "
          << "headers: " << toString(PhdrsOrErr.takeError()) << "\n";
    return std::nullopt;
  }

  if (SawMetadataNote) {
    log() << "hotswap: error: getKernelSgprCount: AMDGPU metadata has no "
          << ".sgpr_count entry for kernel '" << KernelName << "'.\n";
    return std::nullopt;
  }

  // --- Fallback: read the KD field. ---
  // The LLVM assembler populates GRANULATED_WAVEFRONT_SGPR_COUNT even on
  // GFX10+ where the hardware ignores it, so this is still usable for
  // ROCm-compiled code objects that lack a metadata note.
  namespace hsa = amdhsa;
  uint8_t *Kd = const_cast<ElfView *>(this)->findKernelDescriptor(KernelName);
  if (!Kd)
    return std::nullopt;
  uint32_t Rsrc1;
  std::memcpy(&Rsrc1,
              Kd + offsetof(hsa::kernel_descriptor_t, compute_pgm_rsrc1),
              sizeof(Rsrc1));
  uint32_t Granulated = AMDHSA_BITS_GET(
      Rsrc1, hsa::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT);
  uint64_t SgprCount =
      (static_cast<uint64_t>(Granulated) + 1) * SgprEncodingGranule;
  if (SgprCount > std::numeric_limits<unsigned>::max()) {
    log() << "hotswap: error: getKernelSgprCount: descriptor SGPR count for '"
          << KernelName << "' exceeds unsigned.\n";
    return std::nullopt;
  }
  return static_cast<unsigned>(SgprCount);
}

// -- ElfView::updateKernelDescriptor ------------------------------------------

void ElfView::updateKernelDescriptor(StringRef KernelName, unsigned ExtraVgprs,
                                     unsigned VgprGranuleSize) {
  namespace hsa = amdhsa;
  uint8_t *Kd = findKernelDescriptor(KernelName);
  if (!Kd) {
    log() << "hotswap: error: updateKernelDescriptor: kernel descriptor "
          << "symbol '" << KernelName << ".kd' not found; requested +"
          << ExtraVgprs << " VGPRs silently dropped.\n";
    return;
  }

  if (ExtraVgprs == 0 || VgprGranuleSize == 0)
    return;

  uint32_t Rsrc1;
  std::memcpy(&Rsrc1,
              Kd + offsetof(hsa::kernel_descriptor_t, compute_pgm_rsrc1),
              sizeof(Rsrc1));
  uint32_t Current = AMDHSA_BITS_GET(
      Rsrc1, hsa::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT);
  uint32_t MaxGran = static_cast<uint32_t>(
      hsa::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT >>
      hsa::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT_SHIFT);
  uint64_t Extra = (static_cast<uint64_t>(ExtraVgprs) + VgprGranuleSize - 1) /
                   VgprGranuleSize;
  uint64_t NewGranulated =
      std::min<uint64_t>(static_cast<uint64_t>(Current) + Extra, MaxGran);
  AMDHSA_BITS_SET(Rsrc1, hsa::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT,
                  static_cast<uint32_t>(NewGranulated));
  std::memcpy(Kd + offsetof(hsa::kernel_descriptor_t, compute_pgm_rsrc1),
              &Rsrc1, sizeof(Rsrc1));
}

// -- Section/program header adjustment for trampoline growth ------------------

static bool adjustSectionHeaders(uint8_t *Elf, size_t ElfSize,
                                 uint64_t TextOffset, uint64_t TextSize,
                                 size_t TrampTotal) {
  if (ElfSize < sizeof(Ehdr))
    return true;

  std::optional<uint64_t> TextEnd =
      checkedAdd(TextOffset, TextSize, "section header .text end");
  if (!TextEnd)
    return false;
  uint64_t Shoff;
  uint16_t Shentsize;
  uint16_t Shnum;
  std::memcpy(&Shoff, Elf + offsetof(Ehdr, e_shoff), sizeof(Shoff));
  std::memcpy(&Shentsize, Elf + offsetof(Ehdr, e_shentsize), sizeof(Shentsize));
  std::memcpy(&Shnum, Elf + offsetof(Ehdr, e_shnum), sizeof(Shnum));
  if (Shentsize < sizeof(Shdr))
    return true;

  if (Shoff >= *TextEnd) {
    std::optional<uint64_t> NewShoff =
        checkedAdd(Shoff, TrampTotal, "section header table offset");
    if (!NewShoff)
      return false;
    uint64_t NewShoffValue = *NewShoff;
    std::memcpy(Elf + offsetof(Ehdr, e_shoff), &NewShoffValue,
                sizeof(NewShoffValue));
    Shoff = NewShoffValue;
  }

  for (uint16_t I = 0; I < Shnum; ++I) {
    uint64_t ShTableDelta = static_cast<uint64_t>(I) * Shentsize;
    std::optional<uint64_t> ShPos =
        checkedAdd(Shoff, ShTableDelta, "section header entry offset");
    if (!ShPos)
      return false;
    if (*ShPos > ElfSize || sizeof(Shdr) > ElfSize - *ShPos)
      break;
    uint8_t *Sh = Elf + *ShPos;
    uint64_t ShOffset;
    std::memcpy(&ShOffset, Sh + offsetof(Shdr, sh_offset), sizeof(ShOffset));

    if (ShOffset == TextOffset) {
      std::optional<uint64_t> NewTextSize =
          checkedAdd(TextSize, TrampTotal, ".text section size");
      if (!NewTextSize)
        return false;
      uint64_t NewTextSizeValue = *NewTextSize;
      std::memcpy(Sh + offsetof(Shdr, sh_size), &NewTextSizeValue,
                  sizeof(NewTextSizeValue));
    } else if (ShOffset > TextOffset) {
      std::optional<uint64_t> NewOffset =
          checkedAdd(ShOffset, TrampTotal, "post-.text section offset");
      if (!NewOffset)
        return false;
      uint64_t NewOffsetValue = *NewOffset;
      std::memcpy(Sh + offsetof(Shdr, sh_offset), &NewOffsetValue,
                  sizeof(NewOffsetValue));
      uint64_t ShFlags;
      std::memcpy(&ShFlags, Sh + offsetof(Shdr, sh_flags), sizeof(ShFlags));
      if (ShFlags & ELF::SHF_ALLOC) {
        uint64_t ShAddr;
        std::memcpy(&ShAddr, Sh + offsetof(Shdr, sh_addr), sizeof(ShAddr));
        std::optional<uint64_t> NewAddr =
            checkedAdd(ShAddr, TrampTotal, "post-.text section address");
        if (!NewAddr)
          return false;
        ShAddr = *NewAddr;
        std::memcpy(Sh + offsetof(Shdr, sh_addr), &ShAddr, sizeof(ShAddr));
      }
    }
  }
  return true;
}

static bool adjustProgramHeaders(uint8_t *Elf, size_t ElfSize,
                                 uint64_t TextOffset, uint64_t TextSize,
                                 size_t TrampTotal) {
  if (ElfSize < sizeof(Ehdr))
    return true;

  std::optional<uint64_t> TextEnd =
      checkedAdd(TextOffset, TextSize, "program header .text end");
  if (!TextEnd)
    return false;
  uint64_t Phoff;
  uint16_t Phentsize;
  uint16_t Phnum;
  std::memcpy(&Phoff, Elf + offsetof(Ehdr, e_phoff), sizeof(Phoff));
  std::memcpy(&Phentsize, Elf + offsetof(Ehdr, e_phentsize), sizeof(Phentsize));
  std::memcpy(&Phnum, Elf + offsetof(Ehdr, e_phnum), sizeof(Phnum));
  if (Phentsize < sizeof(Phdr))
    return true;

  for (uint16_t I = 0; I < Phnum; ++I) {
    uint64_t PhTableDelta = static_cast<uint64_t>(I) * Phentsize;
    std::optional<uint64_t> PhPos =
        checkedAdd(Phoff, PhTableDelta, "program header entry offset");
    if (!PhPos)
      return false;
    if (*PhPos > ElfSize || sizeof(Phdr) > ElfSize - *PhPos)
      break;
    uint8_t *Ph = Elf + *PhPos;
    uint64_t POffset;
    uint64_t PFilesz;
    uint64_t PMemsz;
    std::memcpy(&POffset, Ph + offsetof(Phdr, p_offset), sizeof(POffset));
    std::memcpy(&PFilesz, Ph + offsetof(Phdr, p_filesz), sizeof(PFilesz));
    std::memcpy(&PMemsz, Ph + offsetof(Phdr, p_memsz), sizeof(PMemsz));

    std::optional<uint64_t> PEnd =
        checkedAdd(POffset, PFilesz, "program header file end");
    if (!PEnd)
      return false;
    if (POffset <= TextOffset && *PEnd >= *TextEnd) {
      std::optional<uint64_t> NewPFilesz =
          checkedAdd(PFilesz, TrampTotal, "program header file size");
      std::optional<uint64_t> NewPMemsz =
          checkedAdd(PMemsz, TrampTotal, "program header memory size");
      if (!NewPFilesz || !NewPMemsz)
        return false;
      PFilesz = *NewPFilesz;
      PMemsz = *NewPMemsz;
      std::memcpy(Ph + offsetof(Phdr, p_filesz), &PFilesz, sizeof(PFilesz));
      std::memcpy(Ph + offsetof(Phdr, p_memsz), &PMemsz, sizeof(PMemsz));
    } else if (POffset > TextOffset) {
      std::optional<uint64_t> NewPOffset =
          checkedAdd(POffset, TrampTotal, "post-.text program offset");
      if (!NewPOffset)
        return false;
      POffset = *NewPOffset;
      std::memcpy(Ph + offsetof(Phdr, p_offset), &POffset, sizeof(POffset));
      uint64_t PVaddr;
      std::memcpy(&PVaddr, Ph + offsetof(Phdr, p_vaddr), sizeof(PVaddr));
      std::optional<uint64_t> NewPVaddr =
          checkedAdd(PVaddr, TrampTotal, "post-.text program vaddr");
      if (!NewPVaddr)
        return false;
      PVaddr = *NewPVaddr;
      std::memcpy(Ph + offsetof(Phdr, p_vaddr), &PVaddr, sizeof(PVaddr));
      uint64_t PPaddr;
      std::memcpy(&PPaddr, Ph + offsetof(Phdr, p_paddr), sizeof(PPaddr));
      std::optional<uint64_t> NewPPaddr =
          checkedAdd(PPaddr, TrampTotal, "post-.text program paddr");
      if (!NewPPaddr)
        return false;
      PPaddr = *NewPPaddr;
      std::memcpy(Ph + offsetof(Phdr, p_paddr), &PPaddr, sizeof(PPaddr));
    }
  }
  return true;
}

static bool adjustSymbolValues(uint8_t *Elf, size_t ElfSize,
                               uint64_t TextOffset, size_t TrampTotal) {
  if (TrampTotal == 0)
    return true;

  Expected<ELFFileT> FileOrErr =
      ELFFileT::create(StringRef(reinterpret_cast<const char *>(Elf), ElfSize));
  if (!FileOrErr) {
    log() << "hotswap: error: adjustSymbolValues: failed to parse grown ELF: "
          << toString(FileOrErr.takeError()) << "\n";
    return false;
  }
  ELFFileT File = std::move(*FileOrErr);

  if (File.getHeader().e_type == ELF::ET_REL)
    return true;

  Expected<ELFT::ShdrRange> SectionsOrErr = File.sections();
  if (!SectionsOrErr) {
    log() << "hotswap: error: adjustSymbolValues: failed to read section "
          << "headers: " << toString(SectionsOrErr.takeError()) << "\n";
    return false;
  }
  ELFT::ShdrRange Sections = *SectionsOrErr;

  unsigned SectionIndex = 0;
  for (const ELFT::Shdr &SymShdr : Sections) {
    if (SymShdr.sh_type != ELF::SHT_SYMTAB &&
        SymShdr.sh_type != ELF::SHT_DYNSYM) {
      ++SectionIndex;
      continue;
    }

    Expected<ELFT::SymRange> SymsOrErr = File.symbols(&SymShdr);
    if (!SymsOrErr) {
      log() << "hotswap: error: adjustSymbolValues: failed to read symbol "
            << "table section " << SectionIndex << ": "
            << toString(SymsOrErr.takeError()) << "\n";
      ++SectionIndex;
      continue;
    }

    for (const ELFT::Sym &Sym : *SymsOrErr) {
      if (Sym.st_shndx == ELF::SHN_UNDEF || Sym.st_shndx >= ELF::SHN_LORESERVE)
        continue;

      Expected<const ELFT::Shdr *> DefShdrOrErr = File.getSection(Sym.st_shndx);
      if (!DefShdrOrErr) {
        log() << "hotswap: error: adjustSymbolValues: symbol references "
              << "missing section " << Sym.st_shndx << ": "
              << toString(DefShdrOrErr.takeError()) << "\n";
        continue;
      }
      const ELFT::Shdr &DefShdr = **DefShdrOrErr;
      if (!(DefShdr.sh_flags & ELF::SHF_ALLOC) ||
          DefShdr.sh_offset <= TextOffset)
        continue;

      const uint8_t *SymBytes = reinterpret_cast<const uint8_t *>(&Sym);
      if (SymBytes < File.base() || SymBytes + sizeof(ELFT::Sym) > File.end()) {
        log() << "hotswap: error: adjustSymbolValues: symbol table entry is "
              << "outside the ELF buffer.\n";
        continue;
      }

      uint64_t SymOffset = SymBytes - File.base();
      std::optional<uint64_t> Value =
          checkedAdd(Sym.st_value, TrampTotal, "post-.text symbol value");
      if (!Value)
        return false;
      uint64_t Value64 = *Value;
      std::memcpy(Elf + SymOffset + offsetof(ELFT::Sym, st_value), &Value64,
                  sizeof(Value64));
    }
    ++SectionIndex;
  }
  return true;
}

// -- ElfView::growWithTrampolines ---------------------------------------------

std::unique_ptr<WritableMemoryBuffer>
ElfView::growWithTrampolines(ArrayRef<Trampoline> Trampolines,
                             ArrayRef<uint8_t> SNopBytes) const {
  const size_t InputSize = size();
  const uint8_t *Input = data();

  size_t TrampTotal = 0;
  for (const Trampoline &T : Trampolines) {
    if (T.Bytes.size() > std::numeric_limits<size_t>::max() - TrampTotal) {
      log() << "hotswap: error: growWithTrampolines: trampoline byte count "
            << "overflows size_t.\n";
      return nullptr;
    }
    TrampTotal += T.Bytes.size();
  }
  if (TrampTotal == 0) {
    log() << "hotswap: growWithTrampolines: no trampolines to insert; "
          << "returning empty result.\n";
    return nullptr;
  }
  if (TrampTotal > std::numeric_limits<size_t>::max() - InputSize) {
    log() << "hotswap: error: growWithTrampolines: trampoline bytes ("
          << TrampTotal << ") + existing ELF size (" << InputSize
          << ") overflow size_t.\n";
    return nullptr;
  }

  std::optional<uint64_t> TextEnd =
      checkedAdd(textOffset(), textSize(), "growWithTrampolines .text end");
  if (!TextEnd || *TextEnd > InputSize) {
    log() << "hotswap: error: growWithTrampolines: .text range exceeds input "
          << "ELF size.\n";
    return nullptr;
  }

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
  std::optional<size_t> PaddedTrampTotalOrErr = checkedAlignToSize(
      TrampTotal, MaxPostTextAlign, "padded trampoline byte count");
  if (!PaddedTrampTotalOrErr)
    return nullptr;
  size_t PaddedTrampTotal = *PaddedTrampTotalOrErr;
  if (PaddedTrampTotal > std::numeric_limits<size_t>::max() - InputSize) {
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
  size_t TextEndSize = static_cast<size_t>(*TextEnd);
  std::memcpy(Out, Input, TextEndSize);
  size_t Pos = TextEndSize;
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
  if (TextEndSize < InputSize)
    std::memcpy(Out + Pos, Input + TextEndSize, InputSize - TextEndSize);

  if (!adjustSectionHeaders(Out, NewSize, textOffset(), textSize(),
                            PaddedTrampTotal) ||
      !adjustProgramHeaders(Out, NewSize, textOffset(), textSize(),
                            PaddedTrampTotal) ||
      !adjustSymbolValues(Out, NewSize, textOffset(), PaddedTrampTotal))
    return nullptr;
  log() << "hotswap: growWithTrampolines: grew ELF from " << InputSize << " to "
        << NewSize << " bytes (" << Trampolines.size() << " trampoline"
        << (Trampolines.size() == 1 ? "" : "s") << ", " << TrampTotal
        << " trampoline bytes + " << PadBytes << " alignment padding).\n";
  return Buf;
}

} // namespace hotswap
} // namespace COMGR
