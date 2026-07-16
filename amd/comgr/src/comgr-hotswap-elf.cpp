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
/// Parses are delegated to llvm::object::ELFFile. ElfView caches immutable
/// symbol ranges and metadata-derived SGPR counts for the duration of one
/// rewrite so large code objects do not repeatedly parse the same ELF data.
///
//===----------------------------------------------------------------------===//

#include "comgr-hotswap-internal.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/BinaryFormat/MsgPackDocument.h"

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

// This file depends on the COMPUTE_PGM_RSRC1_GRANULATED_* field layout below.
// Assert it so the dependency is caught at compile time if it ever shifts.
static_assert(
    amdhsa::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT_SHIFT == 0 &&
        amdhsa::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT_WIDTH == 6,
    "GRANULATED_WORKITEM_VGPR_COUNT layout changed unexpectedly.");
static_assert(
    amdhsa::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT_SHIFT == 6 &&
        amdhsa::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT_WIDTH == 4,
    "GRANULATED_WAVEFRONT_SGPR_COUNT layout changed unexpectedly.");

static constexpr unsigned SgprEncodingGranule = 8;

// Page alignment for the appended trampoline pool's virtual address and file
// offset, so its PT_LOAD segment maps consistently.
static constexpr uint64_t TrampolinePoolAlign = 4096;

enum class MetadataCountUpdateStatus {
  NotFound,
  Found,
  Error,
};

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
  std::optional<uint64_t> FileOffset = checkedAddUint64(
      Sec.sh_offset, Delta, (Twine(Context) + " file offset").str());
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
readUnsignedMetadataNode(const msgpack::DocNode &Node, StringRef KernelName,
                         StringRef Key, StringRef Context) {
  if (Node.getKind() == msgpack::Type::UInt) {
    uint64_t Value = Node.getUInt();
    if (Value > std::numeric_limits<unsigned>::max()) {
      log() << "hotswap: error: " << Context << ": " << Key << " for '"
            << KernelName << "' exceeds unsigned.\n";
      return std::nullopt;
    }
    return static_cast<unsigned>(Value);
  }

  if (Node.getKind() == msgpack::Type::Int) {
    int64_t Value = Node.getInt();
    if (Value < 0 ||
        static_cast<uint64_t>(Value) > std::numeric_limits<unsigned>::max()) {
      log() << "hotswap: error: " << Context << ": " << Key << " for '"
            << KernelName << "' is outside unsigned range.\n";
      return std::nullopt;
    }
    return static_cast<unsigned>(Value);
  }

  log() << "hotswap: error: " << Context << ": " << Key << " for '"
        << KernelName << "' is not an integer.\n";
  return std::nullopt;
}

using MetadataNoteMutator = function_ref<std::optional<bool>(
    msgpack::Document &, msgpack::MapDocNode &)>;
using MetadataNoteValidator = function_ref<bool(bool)>;

struct PendingMetadataWrite {
  size_t Offset = 0;
  std::string Blob;
};

/// Parse each AMDGPU metadata note, invoke \p Mutator on its root map, and
/// defer changed writes until \p Validator accepts the complete traversal.
/// This keeps multi-note updates atomic while sharing the parsing, encoded-size
/// and destination validation for every metadata mutation.
static bool rewriteMetadataNotes(uint8_t *Elf, const ELFFileT &File,
                                 StringRef Context, MetadataNoteMutator Mutator,
                                 MetadataNoteValidator Validator,
                                 bool &SawMetadataNote) {
  Expected<ELFT::PhdrRange> PhdrsOrErr = File.program_headers();
  if (!PhdrsOrErr) {
    log() << "hotswap: error: " << Context << ": failed to read program "
          << "headers: " << toString(PhdrsOrErr.takeError()) << "\n";
    return false;
  }
  std::vector<PendingMetadataWrite> PendingWrites;
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
        log() << "hotswap: error: " << Context
              << ": AMDGPU metadata note has an empty descriptor.\n";
        return false;
      }

      StringRef Blob(reinterpret_cast<const char *>(Desc.data()), Desc.size());
      msgpack::Document Doc;
      if (!Doc.readFromBlob(Blob, false)) {
        log() << "hotswap: error: " << Context
              << ": failed to parse AMDGPU metadata note.\n";
        return false;
      }

      msgpack::DocNode Root = Doc.getRoot();
      if (!Root.isMap()) {
        log() << "hotswap: error: " << Context
              << ": AMDGPU metadata root is not a map.\n";
        return false;
      }

      std::optional<bool> Changed = Mutator(Doc, Root.getMap());
      if (!Changed)
        return false;
      if (!*Changed)
        continue;

      std::string NewBlob;
      Doc.writeToBlob(NewBlob);
      if (NewBlob.size() != Blob.size()) {
        log() << "hotswap: error: " << Context
              << ": updating AMDGPU metadata changes note size from "
              << Blob.size() << " to " << NewBlob.size()
              << " bytes; in-place rewrite cannot preserve ELF layout.\n";
        return false;
      }

      const uint8_t *DescBegin = Desc.data();
      if (DescBegin < File.base() || DescBegin >= File.end()) {
        log() << "hotswap: error: " << Context
              << ": metadata descriptor pointer is outside the ELF buffer.\n";
        return false;
      }
      size_t DescOffset = DescBegin - File.base();
      if (Desc.size() > File.getBufSize() ||
          DescOffset > File.getBufSize() - Desc.size()) {
        log() << "hotswap: error: " << Context
              << ": metadata descriptor extends past the ELF buffer.\n";
        return false;
      }
      PendingWrites.push_back({DescOffset, std::move(NewBlob)});
    }

    if (Err) {
      log() << "hotswap: error: " << Context
            << ": failed to iterate AMDGPU notes: " << toString(std::move(Err))
            << "\n";
      return false;
    }
  }

  if (!Validator(SawMetadataNote))
    return false;
  for (const PendingMetadataWrite &Write : PendingWrites)
    std::memcpy(Elf + Write.Offset, Write.Blob.data(), Write.Blob.size());
  return true;
}

static MetadataCountUpdateStatus
rewriteKernelMetadataCounts(uint8_t *Elf, const ELFFileT &File,
                            const StringMap<unsigned> &RequiredCounts,
                            StringRef MetadataKey, StringRef Context) {
  if (RequiredCounts.empty())
    return MetadataCountUpdateStatus::Found;

  bool SawMetadataNote = false;
  StringMap<bool> Found;
  bool Rewritten = rewriteMetadataNotes(
      Elf, File, Context,
      [&](msgpack::Document &,
          msgpack::MapDocNode &RootMap) -> std::optional<bool> {
        msgpack::DocNode::MapTy::iterator KernelsIt =
            RootMap.find("amdhsa.kernels");
        if (KernelsIt == RootMap.end() || !KernelsIt->second.isArray())
          return false;

        bool Changed = false;
        for (msgpack::DocNode &KNode : KernelsIt->second.getArray()) {
          if (!KNode.isMap())
            continue;
          msgpack::MapDocNode &KMap = KNode.getMap();
          msgpack::DocNode::MapTy::iterator NameIt = KMap.find(".name");
          if (NameIt == KMap.end() || !NameIt->second.isString())
            continue;

          StringRef KernelName = NameIt->second.getString();
          StringMap<unsigned>::const_iterator Required =
              RequiredCounts.find(KernelName);
          if (Required == RequiredCounts.end() || Found.contains(KernelName))
            continue;

          msgpack::DocNode::MapTy::iterator CountIt = KMap.find(MetadataKey);
          if (CountIt == KMap.end()) {
            log() << "hotswap: error: " << Context << ": metadata for kernel '"
                  << KernelName << "' has no " << MetadataKey << ".\n";
            return std::nullopt;
          }

          std::optional<unsigned> CurrentCount = readUnsignedMetadataNode(
              CountIt->second, KernelName, MetadataKey, Context);
          if (!CurrentCount)
            return std::nullopt;
          Found.try_emplace(KernelName, true);
          if (Required->second <= *CurrentCount)
            continue;

          CountIt->second = static_cast<uint64_t>(Required->second);
          Changed = true;
        }
        return Changed;
      },
      [&](bool SawMetadata) {
        if (!SawMetadata)
          return true;
        for (const StringMapEntry<unsigned> &Required : RequiredCounts) {
          if (Found.contains(Required.first()))
            continue;
          log() << "hotswap: error: " << Context
                << ": AMDGPU metadata has no entry for kernel '"
                << Required.first() << "'.\n";
          return false;
        }
        return true;
      },
      SawMetadataNote);
  if (!Rewritten)
    return MetadataCountUpdateStatus::Error;
  return SawMetadataNote ? MetadataCountUpdateStatus::Found
                         : MetadataCountUpdateStatus::NotFound;
}

bool ElfView::updateGfx1250RevisionMetadata(StringRef Revision) {
  bool SawMetadataNote = false;
  return rewriteMetadataNotes(
      data(), File, "updateGfx1250RevisionMetadata",
      [&](msgpack::Document &Doc,
          msgpack::MapDocNode &RootMap) -> std::optional<bool> {
        msgpack::DocNode::MapTy::iterator KernelsIt =
            RootMap.find("amdhsa.kernels");
        if (KernelsIt == RootMap.end() || !KernelsIt->second.isArray())
          return false;

        bool Changed = false;
        for (msgpack::DocNode &KNode : KernelsIt->second.getArray()) {
          if (!KNode.isMap())
            continue;
          msgpack::MapDocNode &KMap = KNode.getMap();
          msgpack::DocNode::MapTy::iterator RevisionIt =
              KMap.find(".gfx1250_revision");
          if (RevisionIt == KMap.end())
            continue;
          if (!RevisionIt->second.isString()) {
            log() << "hotswap: error: updateGfx1250RevisionMetadata: "
                  << ".gfx1250_revision is not a string.\n";
            return std::nullopt;
          }
          if (RevisionIt->second.getString() == Revision)
            continue;
          RevisionIt->second = Doc.getNode(Revision, /*Copy=*/true);
          Changed = true;
        }
        return Changed;
      },
      [](bool) { return true; }, SawMetadataNote);
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
  uint64_t BestDist = std::numeric_limits<uint64_t>::max();
  for (NopSled &Sled : Sleds) {
    if (Offset < Sled.FunctionStart || Offset >= Sled.FunctionEnd)
      continue;
    uint64_t UsableEnd = std::min(Sled.End, Sled.FunctionEnd);
    if (Sled.WritePos > UsableEnd || Needed > UsableEnd - Sled.WritePos)
      continue;
    uint64_t Dist = Sled.WritePos > Offset ? Sled.WritePos - Offset
                                           : Offset - Sled.WritePos;
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

// -- ElfView::functionTextRanges ---------------------------------------------

ArrayRef<ElfView::FunctionTextRange> ElfView::cachedFunctionTextRanges() const {
  if (FunctionRangeCache)
    return *FunctionRangeCache;

  std::vector<FunctionTextRange> Ranges;
  uint64_t TextBegin = textAddr();
  uint64_t TextSizeValue = textSize();
  if (TextSizeValue > std::numeric_limits<uint64_t>::max() - TextBegin) {
    log() << "hotswap: error: function text range scan: .text virtual "
          << "address range overflows uint64_t.\n";
    FunctionRangeCache.emplace();
    return *FunctionRangeCache;
  }
  uint64_t TextEnd = TextBegin + TextSizeValue;

  for (const ELFT::Shdr &SymShdr : Sections) {
    if (SymShdr.sh_type != ELF::SHT_SYMTAB &&
        SymShdr.sh_type != ELF::SHT_DYNSYM)
      continue;

    Expected<ELFT::SymRange> SymsOrErr = File.symbols(&SymShdr);
    if (!SymsOrErr) {
      consumeError(SymsOrErr.takeError());
      continue;
    }

    std::vector<const ELFT::Sym *> FuncSyms;
    for (const ELFT::Sym &Sym : *SymsOrErr) {
      if (Sym.getType() != ELF::STT_FUNC && Sym.getType() != ELF::STT_GNU_IFUNC)
        continue;
      if (Sym.st_shndx != TextSectionIndex)
        continue;
      FuncSyms.push_back(&Sym);
    }
    llvm::sort(FuncSyms, [](const ELFT::Sym *A, const ELFT::Sym *B) {
      if (A->st_value != B->st_value)
        return A->st_value < B->st_value;
      return A->st_size > B->st_size;
    });

    for (size_t I = 0, E = FuncSyms.size(); I != E; ++I) {
      const ELFT::Sym &Sym = *FuncSyms[I];
      uint64_t Begin = Sym.st_value;
      if (Begin < TextBegin || Begin >= TextEnd)
        continue;
      uint64_t End = TextEnd;
      if (Sym.st_size != 0) {
        End = Sym.st_value + Sym.st_size;
        if (End < Begin)
          End = TextEnd;
        End = std::min(End, TextEnd);
      } else {
        for (size_t J = I + 1; J != E; ++J) {
          if (FuncSyms[J]->st_value > Begin) {
            End =
                std::min(static_cast<uint64_t>(FuncSyms[J]->st_value), TextEnd);
            break;
          }
        }
      }
      Ranges.push_back({Begin, End, &Sym, &SymShdr});
    }
  }

  llvm::stable_sort(
      Ranges, [](const FunctionTextRange &LHS, const FunctionTextRange &RHS) {
        return LHS.Begin < RHS.Begin;
      });
  FunctionRangeCache = std::move(Ranges);
  return *FunctionRangeCache;
}

std::vector<ElfView::FunctionTextRange> ElfView::functionTextRanges() const {
  ArrayRef<FunctionTextRange> Ranges = cachedFunctionTextRanges();
  return std::vector<FunctionTextRange>(Ranges.begin(), Ranges.end());
}

// -- ElfView::findKernelAtAddress ---------------------------------------------

const ElfView::FunctionTextRange *
ElfView::findFunctionTextRangeAtAddress(uint64_t TextAddress) const {
  ArrayRef<FunctionTextRange> Ranges = cachedFunctionTextRanges();
  ArrayRef<FunctionTextRange>::const_iterator GroupEnd =
      std::upper_bound(Ranges.begin(), Ranges.end(), TextAddress,
                       [](uint64_t Address, const FunctionTextRange &Range) {
                         return Address < Range.Begin;
                       });

  // Prefer the covering range with the greatest start address, matching the
  // previous full scan. Preserve the stable symbol-table order for duplicate
  // starts so aliases resolve exactly as before.
  while (GroupEnd != Ranges.begin()) {
    ArrayRef<FunctionTextRange>::const_iterator GroupBegin = GroupEnd - 1;
    uint64_t Begin = GroupBegin->Begin;
    while (GroupBegin != Ranges.begin() && (GroupBegin - 1)->Begin == Begin)
      --GroupBegin;
    for (ArrayRef<FunctionTextRange>::const_iterator It = GroupBegin;
         It != GroupEnd; ++It)
      if (TextAddress < It->End)
        return &*It;
    GroupEnd = GroupBegin;
  }
  return nullptr;
}

std::string ElfView::findKernelAtAddress(uint64_t TextAddress) const {
  const FunctionTextRange *Range = findFunctionTextRangeAtAddress(TextAddress);
  if (Range) {
    const ELFT::Sym &Sym = *Range->Symbol;
    Expected<StringRef> StrTabOrErr =
        File.getStringTableForSymtab(*Range->Symtab, Sections);
    if (!StrTabOrErr) {
      consumeError(StrTabOrErr.takeError());
      return "";
    }
    Expected<StringRef> NameOrErr = Sym.getName(*StrTabOrErr);
    if (!NameOrErr) {
      log() << "hotswap: error: findKernelAtAddress: function symbol "
            << "covering address 0x" << utohexstr(TextAddress)
            << " has unreadable name: " << toString(NameOrErr.takeError())
            << "\n";
      return "";
    }
    std::string BestName = NameOrErr->str();
    // Confirm the selected symbol is actually a kernel: every kernel carries a
    // "<name>.kd" descriptor symbol, whereas a plain device function does not.
    // This is the same descriptor lookup getKernelVgprCount performs, so a real
    // kernel is never rejected; a non-kernel is reported as "not found" so the
    // caller declines instead of scratch-allocating against a wrong context.
    if (const_cast<ElfView *>(this)->findKernelDescriptor(BestName)) {
      return BestName;
    }
    log() << "hotswap: findKernelAtAddress: nearest function symbol '"
          << BestName << "' preceding address 0x" << utohexstr(TextAddress)
          << " has no .kd descriptor (not a kernel); treating as no match.\n";
    return "";
  }

  log() << "hotswap: findKernelAtAddress: no function symbol covers address 0x"
        << utohexstr(TextAddress) << " in .text.\n";
  return "";
}

std::optional<ElfView::FunctionTextRange>
ElfView::findFunctionTextRangeAtOffset(uint64_t TextOffset) const {
  if (TextOffset >= textSize() ||
      TextOffset > std::numeric_limits<uint64_t>::max() - textAddr())
    return std::nullopt;
  const FunctionTextRange *Range =
      findFunctionTextRangeAtAddress(textAddr() + TextOffset);
  if (!Range || Range->Begin < textAddr() || Range->End < textAddr())
    return std::nullopt;
  return FunctionTextRange{Range->Begin - textAddr(), Range->End - textAddr(),
                           Range->Symbol, Range->Symtab};
}

// -- ElfView::kernelDescriptors -----------------------------------------------

void ElfView::initializeKernelDescriptorCache() const {
  if (KernelDescriptorCache)
    return;

  namespace hsa = amdhsa;
  std::vector<KernelDescriptorInfo> Result;
  StringMap<uint64_t> FileOffsets;
  StringMap<DenseSet<uint64_t>> SeenVAddr;

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

      StringRef KernelNameRef = NameOrErr->drop_back(3);
      std::string KernelName = KernelNameRef.str();
      // Dedup by (name, vaddr): a kernel name can legitimately map to more than
      // one vaddr, so track the full vaddr set per name rather than only the
      // last one. (The previous per-symbol linear scan over Result made cache
      // init O(n^2).)
      if (SeenVAddr[KernelNameRef].insert(Sym.st_value).second)
        Result.push_back({std::move(KernelName), Sym.st_value, EntryOffset});
      FileOffsets.try_emplace(KernelNameRef, *FileOffset);
    }
  }

  KernelDescriptorFileOffsetCache = std::move(FileOffsets);
  KernelDescriptorCache = std::move(Result);

  // Name -> vaddr map so getKernelDescriptorVAddr() is O(1) per call instead of
  // a linear scan (O(n^2) over ~1000 per-fixup lookups). When a name has more
  // than one descriptor (the dedup set tracks (name, vaddr) pairs), try_emplace
  // keeps the first in symtab order -- the same descriptor the prior linear
  // scan returned, so the single-value lookup is unchanged. The multi-vaddr set
  // matters only for enumeration/dedup, not this name->vaddr resolution.
  KernelDescriptorVAddrCache.clear();
  for (const KernelDescriptorInfo &Info : *KernelDescriptorCache)
    KernelDescriptorVAddrCache.try_emplace(Info.KernelName, Info.VAddr);
}

uint8_t *ElfView::findKernelDescriptor(StringRef KernelName) {
  initializeKernelDescriptorCache();
  StringMap<uint64_t>::const_iterator It =
      KernelDescriptorFileOffsetCache.find(KernelName);
  if (It == KernelDescriptorFileOffsetCache.end())
    return nullptr;
  return data() + It->second;
}

ArrayRef<KernelDescriptorInfo> ElfView::kernelDescriptors() const {
  initializeKernelDescriptorCache();
  return *KernelDescriptorCache;
}

std::optional<uint64_t>
ElfView::getKernelDescriptorVAddr(StringRef KernelName) const {
  initializeKernelDescriptorCache();
  StringMap<uint64_t>::const_iterator It =
      KernelDescriptorVAddrCache.find(KernelName);
  if (It == KernelDescriptorVAddrCache.end())
    return std::nullopt;
  return It->second;
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
  for (KernelDescriptorInfo &Info : *KernelDescriptorCache) {
    if (Info.KernelName != KernelName)
      continue;
    Info.EntryOffset = NewEntryOffset;
    break;
  }
  return true;
}

bool ElfView::updateKernelDescriptorSgprCount(StringRef KernelName,
                                              unsigned RequiredSgprs,
                                              bool UpdateDescriptor) {
  namespace hsa = amdhsa;
  if (RequiredSgprs == 0)
    return true;

  uint8_t *Kd = nullptr;
  uint32_t Rsrc1 = 0;
  std::optional<uint32_t> RequiredGranulated;
  if (UpdateDescriptor) {
    Kd = findKernelDescriptor(KernelName);
    if (!Kd) {
      log() << "hotswap: error: updateKernelDescriptorSgprCount: kernel "
            << "descriptor symbol '" << KernelName << ".kd' not found.\n";
      return false;
    }

    std::memcpy(&Rsrc1,
                Kd + offsetof(hsa::kernel_descriptor_t, compute_pgm_rsrc1),
                sizeof(Rsrc1));

    uint32_t CurrentGranulated = AMDHSA_BITS_GET(
        Rsrc1, hsa::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT);
    uint64_t CurrentSgprs =
        (static_cast<uint64_t>(CurrentGranulated) + 1) * SgprEncodingGranule;

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
  }

  StringMap<unsigned> RequiredSgprCounts;
  RequiredSgprCounts.try_emplace(KernelName, RequiredSgprs);
  MetadataCountUpdateStatus MetadataStatus = rewriteKernelMetadataCounts(
      data(), File, RequiredSgprCounts, ".sgpr_count",
      "updateKernelDescriptorSgprCount");
  if (MetadataStatus == MetadataCountUpdateStatus::Error)
    return false;
  if (!UpdateDescriptor &&
      MetadataStatus == MetadataCountUpdateStatus::NotFound) {
    log() << "hotswap: error: updateKernelDescriptorSgprCount: kernel '"
          << KernelName << "' requires " << RequiredSgprs
          << " SGPRs, but gfx10+ code objects must carry .sgpr_count metadata "
             "because the descriptor SGPR-count field is reserved.\n";
    return false;
  }
  // On pre-gfx10 targets, NotFound is allowed for minimal code objects without
  // AMDGPU metadata because the descriptor remains the canonical count.

  if (SgprCacheState == KernelSgprCacheState::Metadata &&
      MetadataStatus == MetadataCountUpdateStatus::Found) {
    StringMap<std::optional<unsigned>>::iterator Cached =
        KernelSgprCountCache.find(KernelName);
    if (Cached == KernelSgprCountCache.end())
      KernelSgprCountCache.try_emplace(KernelName, RequiredSgprs);
    else if (!Cached->second || RequiredSgprs > *Cached->second)
      Cached->second = RequiredSgprs;
  }

  if (!RequiredGranulated)
    return true;

  AMDHSA_BITS_SET(Rsrc1, hsa::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                  *RequiredGranulated);
  std::memcpy(Kd + offsetof(hsa::kernel_descriptor_t, compute_pgm_rsrc1),
              &Rsrc1, sizeof(Rsrc1));
  if (SgprCacheState == KernelSgprCacheState::NoMetadata)
    KernelSgprCountCache.erase(KernelName);
  return true;
}

bool ElfView::updateKernelMetadataSgprCounts(
    const StringMap<unsigned> &RequiredSgprs) {
  MetadataCountUpdateStatus MetadataStatus =
      rewriteKernelMetadataCounts(data(), File, RequiredSgprs, ".sgpr_count",
                                  "updateKernelMetadataSgprCounts");
  if (MetadataStatus == MetadataCountUpdateStatus::Error)
    return false;
  if (MetadataStatus == MetadataCountUpdateStatus::NotFound) {
    log() << "hotswap: error: updateKernelMetadataSgprCounts: code object "
             "has no AMDGPU metadata note.\n";
    return false;
  }

  if (SgprCacheState == KernelSgprCacheState::Metadata) {
    for (const StringMapEntry<unsigned> &Required : RequiredSgprs) {
      StringMap<std::optional<unsigned>>::iterator Cached =
          KernelSgprCountCache.find(Required.first());
      if (Cached == KernelSgprCountCache.end())
        KernelSgprCountCache.try_emplace(Required.first(), Required.second);
      else if (!Cached->second || Required.second > *Cached->second)
        Cached->second = Required.second;
    }
  }
  return true;
}

bool ElfView::updateKernelMetadataVgprCounts(
    const StringMap<unsigned> &RequiredVgprs) {
  MetadataCountUpdateStatus MetadataStatus =
      rewriteKernelMetadataCounts(data(), File, RequiredVgprs, ".vgpr_count",
                                  "updateKernelMetadataVgprCounts");
  if (MetadataStatus == MetadataCountUpdateStatus::Error)
    return false;
  if (MetadataStatus == MetadataCountUpdateStatus::NotFound) {
    log() << "hotswap: error: updateKernelMetadataVgprCounts: code object "
             "has no AMDGPU metadata note.\n";
    return false;
  }
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

bool ElfView::updateKernelDescriptorInstPrefSize(StringRef KernelName,
                                                 StringRef TargetCpu,
                                                 uint32_t InstPrefLines) {
  namespace hsa = amdhsa;
  uint8_t *Kd = findKernelDescriptor(KernelName);
  if (!Kd) {
    log() << "hotswap: error: updateKernelDescriptorInstPrefSize: kernel "
          << "descriptor symbol '" << KernelName << ".kd' not found.\n";
    return false;
  }

  if (!TargetCpu.starts_with("gfx12")) {
    log() << "hotswap: error: updateKernelDescriptorInstPrefSize: unsupported "
          << "target CPU '" << TargetCpu << "' for kernel '" << KernelName
          << "'.\n";
    return false;
  }

  uint32_t MaxInstPrefLines = static_cast<uint32_t>(
      hsa::COMPUTE_PGM_RSRC3_GFX12_PLUS_INST_PREF_SIZE >>
      hsa::COMPUTE_PGM_RSRC3_GFX12_PLUS_INST_PREF_SIZE_SHIFT);
  if (InstPrefLines > MaxInstPrefLines) {
    log() << "hotswap: error: updateKernelDescriptorInstPrefSize: value "
          << InstPrefLines << " exceeds the gfx12 descriptor encoding limit.\n";
    return false;
  }

  uint32_t Rsrc3 = 0;
  std::memcpy(&Rsrc3,
              Kd + offsetof(hsa::kernel_descriptor_t, compute_pgm_rsrc3),
              sizeof(Rsrc3));
  AMDHSA_BITS_SET(Rsrc3, hsa::COMPUTE_PGM_RSRC3_GFX12_PLUS_INST_PREF_SIZE,
                  InstPrefLines);
  std::memcpy(Kd + offsetof(hsa::kernel_descriptor_t, compute_pgm_rsrc3),
              &Rsrc3, sizeof(Rsrc3));
  return true;
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
  // call path but is shared (non-const) with descriptor update helpers. The
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

static std::optional<unsigned> getKernelUnsignedMetadata(const ELFFileT &File,
                                                         StringRef KernelName,
                                                         StringRef Key,
                                                         StringRef Context) {
  Expected<ELFT::PhdrRange> PhdrsOrErr = File.program_headers();
  if (!PhdrsOrErr) {
    log() << "hotswap: error: " << Context
          << ": failed to read program headers: "
          << toString(PhdrsOrErr.takeError()) << "\n";
    return std::nullopt;
  }

  for (const ELFT::Phdr &Phdr : *PhdrsOrErr) {
    if (Phdr.p_type != ELF::PT_NOTE)
      continue;

    Error Err = Error::success();
    for (ELFT::Note Note : File.notes(Phdr, Err)) {
      if (Note.getName() != "AMDGPU" ||
          Note.getType() != ELF::NT_AMDGPU_METADATA)
        continue;

      ArrayRef<uint8_t> Desc = Note.getDesc(4);
      if (Desc.empty()) {
        log() << "hotswap: error: " << Context
              << ": AMDGPU metadata note has an empty descriptor.\n";
        return std::nullopt;
      }

      StringRef Blob(reinterpret_cast<const char *>(Desc.data()), Desc.size());
      msgpack::Document Doc;
      if (!Doc.readFromBlob(Blob, false)) {
        log() << "hotswap: error: " << Context
              << ": failed to parse AMDGPU metadata note.\n";
        return std::nullopt;
      }

      msgpack::DocNode Root = Doc.getRoot();
      if (!Root.isMap()) {
        log() << "hotswap: error: " << Context
              << ": AMDGPU metadata root is not a map.\n";
        return std::nullopt;
      }

      msgpack::MapDocNode &RootMap = Root.getMap();
      msgpack::DocNode::MapTy::iterator KernelsIt =
          RootMap.find("amdhsa.kernels");
      if (KernelsIt == RootMap.end() || !KernelsIt->second.isArray())
        continue;

      msgpack::ArrayDocNode &KernelArray = KernelsIt->second.getArray();
      for (msgpack::DocNode &KernelNode : KernelArray) {
        if (!KernelNode.isMap())
          continue;
        msgpack::MapDocNode &KernelMap = KernelNode.getMap();
        msgpack::DocNode::MapTy::iterator NameIt = KernelMap.find(".name");
        if (NameIt == KernelMap.end() || !NameIt->second.isString() ||
            NameIt->second.getString() != KernelName)
          continue;

        msgpack::DocNode::MapTy::iterator ValueIt = KernelMap.find(Key);
        if (ValueIt == KernelMap.end())
          return std::nullopt;
        return readUnsignedMetadataNode(ValueIt->second, KernelName, Key,
                                        Context);
      }
    }

    if (Err) {
      log() << "hotswap: error: " << Context
            << ": failed to iterate AMDGPU notes: " << toString(std::move(Err))
            << "\n";
      return std::nullopt;
    }
  }
  return std::nullopt;
}

std::optional<unsigned>
ElfView::getKernelMaxFlatWorkgroupSize(StringRef KernelName) const {
  return getKernelUnsignedMetadata(File, KernelName, ".max_flat_workgroup_size",
                                   "getKernelMaxFlatWorkgroupSize");
}

std::optional<unsigned>
ElfView::getKernelMetadataVgprCount(StringRef KernelName) const {
  return getKernelUnsignedMetadata(File, KernelName, ".vgpr_count",
                                   "getKernelMetadataVgprCount");
}

std::optional<unsigned>
ElfView::getKernelWavefrontSize(StringRef KernelName) const {
  return getKernelUnsignedMetadata(File, KernelName, ".wavefront_size",
                                   "getKernelWavefrontSize");
}

// Reads the static (compile-time-fixed) LDS allocation from the kernel
// descriptor's group_segment_fixed_size field. Dynamic LDS is added by the
// host at dispatch time and is not visible here -- see the declaration's
// doc comment for the full lower-bound caveat.

std::optional<uint32_t>
ElfView::getKernelStaticLdsSize(StringRef KernelName) const {
  namespace hsa = amdhsa;
  // findKernelDescriptor never writes through the returned pointer in this
  // call path but is shared (non-const) with descriptor update helpers. The
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

void ElfView::initializeKernelSgprCountCache() const {
  if (SgprCacheState != KernelSgprCacheState::Uninitialized)
    return;

  // Default to Error so every malformed-note early return leaves an explicit
  // terminal cache state instead of reparsing the same large blob.
  SgprCacheState = KernelSgprCacheState::Error;
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
          log() << "hotswap: error: SGPR cache: AMDGPU metadata note "
                << "has an empty descriptor.\n";
          return;
        }

        StringRef Blob(reinterpret_cast<const char *>(Desc.data()),
                       Desc.size());
        msgpack::Document Doc;
        if (!Doc.readFromBlob(Blob, false)) {
          log() << "hotswap: error: SGPR cache: failed to parse "
                << "AMDGPU metadata note.\n";
          return;
        }

        msgpack::DocNode Root = Doc.getRoot();
        if (!Root.isMap()) {
          log() << "hotswap: error: SGPR cache: AMDGPU metadata root "
                << "is not a map.\n";
          return;
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
              KernelSgprCountCache.find(NameIt->second.getString()) !=
                  KernelSgprCountCache.end())
            continue;

          msgpack::DocNode::MapTy::iterator SgprIt = KMap.find(".sgpr_count");
          if (SgprIt == KMap.end()) {
            KernelSgprCountCache.try_emplace(NameIt->second.getString(),
                                             std::nullopt);
            continue;
          }
          StringRef Name = NameIt->second.getString();
          KernelSgprCountCache.try_emplace(
              Name, readUnsignedMetadataNode(SgprIt->second, Name,
                                             ".sgpr_count", "SGPR cache"));
        }
      }
      if (Err) {
        log() << "hotswap: error: SGPR cache: failed to iterate "
              << "AMDGPU notes: " << toString(std::move(Err)) << "\n";
        return;
      }
    }
  } else {
    log() << "hotswap: error: SGPR cache: failed to read program "
          << "headers: " << toString(PhdrsOrErr.takeError()) << "\n";
    return;
  }

  SgprCacheState = SawMetadataNote ? KernelSgprCacheState::Metadata
                                   : KernelSgprCacheState::NoMetadata;
}

std::optional<unsigned>
ElfView::getKernelSgprCount(StringRef KernelName) const {
  initializeKernelSgprCountCache();
  if (SgprCacheState == KernelSgprCacheState::Error)
    return std::nullopt;

  StringMap<std::optional<unsigned>>::const_iterator Cached =
      KernelSgprCountCache.find(KernelName);
  if (SgprCacheState == KernelSgprCacheState::Metadata) {
    if (Cached != KernelSgprCountCache.end()) {
      if (!Cached->second)
        log() << "hotswap: error: getKernelSgprCount: metadata for kernel '"
              << KernelName << "' has no valid .sgpr_count.\n";
      return Cached->second;
    }
    log() << "hotswap: error: getKernelSgprCount: AMDGPU metadata has no "
          << ".sgpr_count entry for kernel '" << KernelName << "'.\n";
    return std::nullopt;
  }

  if (Cached != KernelSgprCountCache.end())
    return Cached->second;

  // --- Fallback: read the KD field. ---
  // The LLVM assembler populates GRANULATED_WAVEFRONT_SGPR_COUNT even on
  // GFX10+ where the hardware ignores it, so this is still usable for
  // ROCm-compiled code objects that lack a metadata note.
  namespace hsa = amdhsa;
  uint8_t *Kd = const_cast<ElfView *>(this)->findKernelDescriptor(KernelName);
  if (!Kd) {
    KernelSgprCountCache.try_emplace(KernelName, std::nullopt);
    return std::nullopt;
  }
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
  std::optional<unsigned> Result = static_cast<unsigned>(SgprCount);
  KernelSgprCountCache.try_emplace(KernelName, Result);
  return Result;
}

// -- ElfView::getKernelClusterDims --------------------------------------------
//
// Reads optional fixed .cluster_dims metadata from the amdhsa.kernels msgpack
// note. Absence is expected for kernels with variable dispatch-time cluster
// dimensions, so callers use std::nullopt as the dynamic fallback signal.

std::optional<KernelClusterDims>
ElfView::getKernelClusterDims(StringRef KernelName) const {
  Expected<ELFT::PhdrRange> PhdrsOrErr = File.program_headers();
  if (!PhdrsOrErr) {
    log() << "hotswap: error: getKernelClusterDims: failed to read program "
          << "headers: " << toString(PhdrsOrErr.takeError()) << "\n";
    return std::nullopt;
  }

  for (const ELFT::Phdr &Phdr : *PhdrsOrErr) {
    if (Phdr.p_type != ELF::PT_NOTE)
      continue;

    Error Err = Error::success();
    for (ELFT::Note Note : File.notes(Phdr, Err)) {
      if (Note.getName() != "AMDGPU" ||
          Note.getType() != ELF::NT_AMDGPU_METADATA)
        continue;

      ArrayRef<uint8_t> Desc = Note.getDesc(4);
      if (Desc.empty()) {
        log() << "hotswap: error: getKernelClusterDims: AMDGPU metadata note "
              << "has an empty descriptor.\n";
        return std::nullopt;
      }

      StringRef Blob(reinterpret_cast<const char *>(Desc.data()), Desc.size());
      msgpack::Document Doc;
      if (!Doc.readFromBlob(Blob, false)) {
        log() << "hotswap: error: getKernelClusterDims: failed to parse "
              << "AMDGPU metadata note.\n";
        return std::nullopt;
      }

      msgpack::DocNode Root = Doc.getRoot();
      if (!Root.isMap()) {
        log() << "hotswap: error: getKernelClusterDims: AMDGPU metadata root "
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

        msgpack::DocNode::MapTy::iterator DimsIt = KMap.find(".cluster_dims");
        if (DimsIt == KMap.end())
          return std::nullopt;
        if (!DimsIt->second.isArray()) {
          log() << "hotswap: error: getKernelClusterDims: .cluster_dims for '"
                << KernelName << "' is not an array.\n";
          return std::nullopt;
        }

        msgpack::ArrayDocNode &Dims = DimsIt->second.getArray();
        if (Dims.size() != 3) {
          log() << "hotswap: error: getKernelClusterDims: .cluster_dims for '"
                << KernelName << "' has " << Dims.size()
                << " entries, expected 3.\n";
          return std::nullopt;
        }

        std::optional<unsigned> X = readUnsignedMetadataNode(
            Dims[0], KernelName, ".cluster_dims[0]", "getKernelClusterDims");
        std::optional<unsigned> Y = readUnsignedMetadataNode(
            Dims[1], KernelName, ".cluster_dims[1]", "getKernelClusterDims");
        std::optional<unsigned> Z = readUnsignedMetadataNode(
            Dims[2], KernelName, ".cluster_dims[2]", "getKernelClusterDims");
        if (!X || !Y || !Z)
          return std::nullopt;
        return KernelClusterDims{*X, *Y, *Z};
      }
    }

    if (Err) {
      log() << "hotswap: error: getKernelClusterDims: failed to iterate "
            << "AMDGPU notes: " << toString(std::move(Err)) << "\n";
      return std::nullopt;
    }
  }

  return std::nullopt;
}

// -- ElfView::updateKernelDescriptorVgprCount ---------------------------------

bool ElfView::updateKernelDescriptorVgprCount(StringRef KernelName,
                                              unsigned RequiredVgprs,
                                              unsigned VgprGranuleSize) {
  namespace hsa = amdhsa;
  if (VgprGranuleSize == 0) {
    log() << "hotswap: error: updateKernelDescriptorVgprCount: VGPR granule "
             "is zero for kernel '"
          << KernelName << "'.\n";
    return false;
  }

  uint8_t *Kd = findKernelDescriptor(KernelName);
  if (!Kd) {
    log() << "hotswap: error: updateKernelDescriptorVgprCount: kernel "
             "descriptor symbol '"
          << KernelName << ".kd' not found.\n";
    return false;
  }

  uint32_t Rsrc1;
  std::memcpy(&Rsrc1,
              Kd + offsetof(hsa::kernel_descriptor_t, compute_pgm_rsrc1),
              sizeof(Rsrc1));
  uint32_t CurrentGranulated = AMDHSA_BITS_GET(
      Rsrc1, hsa::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT);
  uint64_t CurrentVgprs =
      (static_cast<uint64_t>(CurrentGranulated) + 1) * VgprGranuleSize;
  if (RequiredVgprs <= CurrentVgprs)
    return true;

  uint32_t MaxGran = static_cast<uint32_t>(
      hsa::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT >>
      hsa::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT_SHIFT);
  uint64_t NewGranulated =
      (static_cast<uint64_t>(RequiredVgprs) + VgprGranuleSize - 1) /
          VgprGranuleSize -
      1;
  if (NewGranulated > MaxGran) {
    log() << "hotswap: error: updateKernelDescriptorVgprCount: kernel '"
          << KernelName << "' needs " << RequiredVgprs
          << " VGPRs, which exceeds the descriptor encoding limit.\n";
    return false;
  }
  AMDHSA_BITS_SET(Rsrc1, hsa::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT,
                  static_cast<uint32_t>(NewGranulated));
  std::memcpy(Kd + offsetof(hsa::kernel_descriptor_t, compute_pgm_rsrc1),
              &Rsrc1, sizeof(Rsrc1));
  return true;
}

// -- ElfView::dataAtVAddr -----------------------------------------------------

const uint8_t *ElfView::dataAtVAddr(uint64_t VAddr, uint64_t Len) const {
  for (const ELFT::Shdr &Shdr : Sections) {
    if (!(Shdr.sh_flags & ELF::SHF_ALLOC) || Shdr.sh_type == ELF::SHT_NOBITS)
      continue;
    if (VAddr < Shdr.sh_addr)
      continue;
    uint64_t Off = VAddr - Shdr.sh_addr;
    if (Off > Shdr.sh_size || Len > Shdr.sh_size - Off)
      continue;
    if (Shdr.sh_offset > size() || Off > size() - Shdr.sh_offset ||
        Len > size() - Shdr.sh_offset - Off)
      continue;
    return data() + Shdr.sh_offset + Off;
  }
  return nullptr;
}

// -- ElfView::trampolinePoolVAddr ---------------------------------------------

std::optional<uint64_t> ElfView::trampolinePoolVAddr() const {
  uint64_t MaxAllocEnd = 0;
  for (const ELFT::Shdr &Shdr : Sections) {
    if (!(Shdr.sh_flags & ELF::SHF_ALLOC))
      continue;
    // Overflow would collapse MaxAllocEnd and overlap the pool with existing
    // sections.
    std::optional<uint64_t> End = checkedAddUint64(
        Shdr.sh_addr, Shdr.sh_size, "allocatable section end for pool vaddr");
    if (!End)
      return std::nullopt;
    MaxAllocEnd = std::max(MaxAllocEnd, *End);
  }
  return alignTo(MaxAllocEnd, TrampolinePoolAlign);
}

// -- addKernelEntryTrampolineSymbols ------------------------------------------

std::unique_ptr<WritableMemoryBuffer> addKernelEntryTrampolineSymbols(
    WritableMemoryBuffer &In, unsigned TextSectionIndex, uint64_t TextAddr,
    uint64_t OldTextSize, ArrayRef<KernelEntryTrampolineFixup> Fixups) {
  if (Fixups.empty())
    return nullptr;

  const uint8_t *Data = reinterpret_cast<const uint8_t *>(In.getBufferStart());
  const size_t Size = In.getBufferSize();

  Expected<ELFFileT> FileOrErr =
      ELFFileT::create(StringRef(reinterpret_cast<const char *>(Data), Size));
  if (!FileOrErr) {
    log() << "hotswap: error: addKernelEntryTrampolineSymbols: failed to parse "
          << "grown ELF: " << toString(FileOrErr.takeError()) << "\n";
    return nullptr;
  }
  ELFFileT File = std::move(*FileOrErr);
  Expected<ELFT::ShdrRange> SecsOrErr = File.sections();
  if (!SecsOrErr) {
    consumeError(SecsOrErr.takeError());
    return nullptr;
  }
  ELFT::ShdrRange Secs = *SecsOrErr;

  // Locate .symtab and its linked string table. Scan from the end, since the
  // symbol table sits near the end of the section list in these code objects.
  const ELFT::Shdr *SymShdr = nullptr;
  unsigned SymIdx = 0;
  for (unsigned I = Secs.size(); I-- > 0;)
    if (Secs[I].sh_type == ELF::SHT_SYMTAB) {
      SymShdr = &Secs[I];
      SymIdx = I;
      break;
    }
  if (!SymShdr) {
    log() << "hotswap: addKernelEntryTrampolineSymbols: no .symtab present; "
          << "skipping stub symbols.\n";
    return nullptr;
  }
  const unsigned StrIdx = SymShdr->sh_link;
  if (StrIdx == 0 || StrIdx >= Secs.size()) {
    log() << "hotswap: error: addKernelEntryTrampolineSymbols: .symtab has an "
          << "invalid sh_link (" << StrIdx << ").\n";
    return nullptr;
  }
  if (SymShdr->sh_entsize != sizeof(ELF::Elf64_Sym)) {
    log() << "hotswap: error: addKernelEntryTrampolineSymbols: unexpected "
          << ".symtab entry size " << SymShdr->sh_entsize << ".\n";
    return nullptr;
  }
  const ELFT::Shdr &StrShdr = Secs[StrIdx];

  const uint64_t SymOff = SymShdr->sh_offset;
  const uint64_t SymEnd = SymOff + SymShdr->sh_size;
  const uint64_t StrOff = StrShdr.sh_offset;
  const uint64_t StrEnd = StrOff + StrShdr.sh_size;
  if (SymEnd > Size || StrEnd > Size || SymEnd < SymOff || StrEnd < StrOff) {
    log() << "hotswap: error: addKernelEntryTrampolineSymbols: symbol/string "
          << "table extends past the ELF buffer.\n";
    return nullptr;
  }

  // Build the appended string names and symbol entries.
  SmallVector<uint8_t> StrBlob, SymBlob;
  for (const KernelEntryTrampolineFixup &F : Fixups) {
    std::string Name = F.KernelName + ".stub";
    uint32_t NameOff = static_cast<uint32_t>(StrShdr.sh_size + StrBlob.size());
    StrBlob.append(Name.begin(), Name.end());
    StrBlob.push_back(0);

    std::optional<uint64_t> StubOff = checkedAddUint64(
        OldTextSize, F.StubTextOffset, "stub symbol .text offset");
    if (!StubOff)
      return nullptr;
    std::optional<uint64_t> StubVAddr =
        checkedAddUint64(TextAddr, *StubOff, "stub symbol vaddr");
    if (!StubVAddr)
      return nullptr;

    ELF::Elf64_Sym Sym{};
    Sym.st_name = NameOff;
    Sym.st_info = (ELF::STB_GLOBAL << 4) | ELF::STT_FUNC;
    Sym.st_other = ELF::STV_DEFAULT;
    Sym.st_shndx = static_cast<uint16_t>(TextSectionIndex);
    Sym.st_value = *StubVAddr;
    Sym.st_size = KernelEntryStubStride;
    const uint8_t *P = reinterpret_cast<const uint8_t *>(&Sym);
    SymBlob.append(P, P + sizeof(Sym));
  }
  // The section header table must stay 8-byte aligned (LLVM's ELF reader
  // rejects a misaligned table). SymBlob is a multiple of 8 (24-byte entries),
  // so pad the string blob up to a multiple of 8 with unreferenced NULs.
  StrBlob.append((8 - (StrBlob.size() % 8)) % 8, 0);

  const uint64_t SymDelta = SymBlob.size();
  const uint64_t StrDelta = StrBlob.size();
  const size_t NewSize = Size + SymDelta + StrDelta;

  std::unique_ptr<WritableMemoryBuffer> Out =
      WritableMemoryBuffer::getNewUninitMemBuffer(NewSize);
  if (!Out) {
    log() << "hotswap: error: addKernelEntryTrampolineSymbols: allocation of "
          << NewSize << " bytes failed.\n";
    return nullptr;
  }
  uint8_t *O = reinterpret_cast<uint8_t *>(Out->getBufferStart());

  // Insert the new symbol entries right after the existing .symtab contents and
  // the new strings right after the existing .strtab contents. Both insertion
  // points are expressed in original-file coordinates; copying in ascending
  // order keeps the arithmetic order-independent (either table may come first).
  struct Insertion {
    uint64_t Pos;
    const SmallVector<uint8_t> *Bytes;
  };
  Insertion A{SymEnd, &SymBlob}, B{StrEnd, &StrBlob};
  if (A.Pos > B.Pos)
    std::swap(A, B);

  size_t OutPos = 0, InPos = 0;
  auto CopyThrough = [&](uint64_t Upto) {
    std::memcpy(O + OutPos, Data + InPos, Upto - InPos);
    OutPos += Upto - InPos;
    InPos = Upto;
  };
  CopyThrough(A.Pos);
  std::memcpy(O + OutPos, A.Bytes->data(), A.Bytes->size());
  OutPos += A.Bytes->size();
  CopyThrough(B.Pos);
  std::memcpy(O + OutPos, B.Bytes->data(), B.Bytes->size());
  OutPos += B.Bytes->size();
  std::memcpy(O + OutPos, Data + InPos, Size - InPos);

  // Anything at or beyond an insertion point shifts by that insertion's size.
  auto Shift = [&](uint64_t X) -> uint64_t {
    return X + (X >= SymEnd ? SymDelta : 0) + (X >= StrEnd ? StrDelta : 0);
  };

  uint64_t Shoff;
  uint16_t Shentsize, Shnum;
  std::memcpy(&Shoff, O + offsetof(Ehdr, e_shoff), sizeof(Shoff));
  std::memcpy(&Shentsize, O + offsetof(Ehdr, e_shentsize), sizeof(Shentsize));
  std::memcpy(&Shnum, O + offsetof(Ehdr, e_shnum), sizeof(Shnum));

  uint64_t Phoff;
  uint16_t Phentsize, Phnum;
  std::memcpy(&Phoff, O + offsetof(Ehdr, e_phoff), sizeof(Phoff));
  std::memcpy(&Phentsize, O + offsetof(Ehdr, e_phentsize), sizeof(Phentsize));
  std::memcpy(&Phnum, O + offsetof(Ehdr, e_phnum), sizeof(Phnum));

  uint64_t NewShoff = Shift(Shoff);
  std::memcpy(O + offsetof(Ehdr, e_shoff), &NewShoff, sizeof(NewShoff));
  uint64_t NewPhoff = Shift(Phoff);
  std::memcpy(O + offsetof(Ehdr, e_phoff), &NewPhoff, sizeof(NewPhoff));

  if (Shentsize < sizeof(Shdr))
    return nullptr;

  for (uint16_t I = 0; I < Shnum; ++I) {
    uint64_t P = NewShoff + static_cast<uint64_t>(I) * Shentsize;
    if (P + sizeof(Shdr) > NewSize)
      break;
    uint8_t *Sh = O + P;
    uint64_t ShOffset;
    std::memcpy(&ShOffset, Sh + offsetof(Shdr, sh_offset), sizeof(ShOffset));
    uint64_t NewOff = Shift(ShOffset);
    std::memcpy(Sh + offsetof(Shdr, sh_offset), &NewOff, sizeof(NewOff));

    if (I == SymIdx || I == StrIdx) {
      uint64_t ShSize;
      std::memcpy(&ShSize, Sh + offsetof(Shdr, sh_size), sizeof(ShSize));
      ShSize += (I == SymIdx) ? SymDelta : StrDelta;
      std::memcpy(Sh + offsetof(Shdr, sh_size), &ShSize, sizeof(ShSize));
    }
  }

  if (Phentsize >= sizeof(Phdr)) {
    for (uint16_t I = 0; I < Phnum; ++I) {
      uint64_t P = NewPhoff + static_cast<uint64_t>(I) * Phentsize;
      if (P + sizeof(Phdr) > NewSize)
        break;
      uint8_t *Ph = O + P;
      uint64_t POffset;
      std::memcpy(&POffset, Ph + offsetof(Phdr, p_offset), sizeof(POffset));
      uint64_t NewPOffset = Shift(POffset);
      std::memcpy(Ph + offsetof(Phdr, p_offset), &NewPOffset,
                  sizeof(NewPOffset));
    }
  }

  log() << "hotswap: added " << Fixups.size()
        << " kernel-entry stub symbol(s) to .symtab\n";
  return Out;
}

// -- ElfView::growWithTrampolines ---------------------------------------------

std::unique_ptr<WritableMemoryBuffer>
ElfView::growWithTrampolines(ArrayRef<Trampoline> Trampolines,
                             ArrayRef<uint8_t> SNopBytes) const {
  // SNopBytes is unused in the append-at-end model: nothing between .text and
  // the following sections moves, so there is no in-image gap to pad. It is
  // retained in the signature for callers and for a future in-place variant.
  (void)SNopBytes;

  const size_t InputSize = size();
  const uint8_t *Input = data();

  if (InputSize < sizeof(Ehdr)) {
    log() << "hotswap: error: growWithTrampolines: input (" << InputSize
          << " bytes) is smaller than an ELF64 header.\n";
    return nullptr;
  }

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

  // Append the pool at a fresh virtual address above every existing
  // allocatable section (trampolinePoolVAddr()). Because existing sections,
  // symbols, and program headers keep their addresses, the baked PC-relative
  // literals (and DWARF) that reference post-.text data stay valid. The
  // previous scheme grew .text in place and shifted everything after it,
  // silently corrupting those baked references (a fully-linked AMDGPU object
  // carries no relocations) -- see
  // ElfView.GrowWithTrampolinesKeepsIsaReferenceConsistentWithSymbol.
  //
  // The vaddr and file offset are page-aligned (equal modulo the alignment) so
  // the appended PT_LOAD maps consistently.
  std::optional<uint64_t> PoolVAddrOr = trampolinePoolVAddr();
  if (!PoolVAddrOr) {
    log() << "hotswap: error: growWithTrampolines: could not compute a "
          << "trampoline pool virtual address.\n";
    return nullptr;
  }
  const uint64_t PoolVAddr = *PoolVAddrOr;
  const uint64_t PoolFileOff =
      alignTo(static_cast<uint64_t>(InputSize), TrampolinePoolAlign);

  // Copy the program-header and section-header tables to the end of the file,
  // each with one new entry for the pool (a PT_LOAD segment so the loader maps
  // it, and an SHF_ALLOC|SHF_EXECINSTR section so objdump/tools and a
  // subsequent rewrite can see it), then repoint e_phoff / e_shoff. Those
  // tables are metadata addressed via the ELF header, so relocating them moves
  // nothing a baked literal can reference.
  uint64_t Phoff, Shoff;
  uint16_t Phentsize, Phnum, Shentsize, Shnum;
  std::memcpy(&Phoff, Input + offsetof(Ehdr, e_phoff), sizeof(Phoff));
  std::memcpy(&Phentsize, Input + offsetof(Ehdr, e_phentsize),
              sizeof(Phentsize));
  std::memcpy(&Phnum, Input + offsetof(Ehdr, e_phnum), sizeof(Phnum));
  std::memcpy(&Shoff, Input + offsetof(Ehdr, e_shoff), sizeof(Shoff));
  std::memcpy(&Shentsize, Input + offsetof(Ehdr, e_shentsize),
              sizeof(Shentsize));
  std::memcpy(&Shnum, Input + offsetof(Ehdr, e_shnum), sizeof(Shnum));

  const bool HasPhdrs =
      Phnum > 0 && Phoff != 0 && Phentsize >= sizeof(Phdr) &&
      Phoff <= InputSize &&
      static_cast<uint64_t>(Phnum) * Phentsize <= InputSize - Phoff;
  const bool HasShdrs =
      Shnum > 0 && Shoff != 0 && Shentsize >= sizeof(Shdr) &&
      Shoff <= InputSize &&
      static_cast<uint64_t>(Shnum) * Shentsize <= InputSize - Shoff;

  std::optional<uint64_t> PoolEnd =
      checkedAddUint64(PoolFileOff, TrampTotal, "trampoline pool file end");
  if (!PoolEnd)
    return nullptr;

  // Lay out the relocated tables after the pool: [pool][phdrs][shdrs].
  uint64_t Cursor = *PoolEnd;
  const uint64_t NewPhnum = HasPhdrs ? static_cast<uint64_t>(Phnum) + 1 : Phnum;
  const uint64_t NewShnum = HasShdrs ? static_cast<uint64_t>(Shnum) + 1 : Shnum;
  uint64_t NewPhoff = Phoff;
  uint64_t NewShoff = Shoff;
  if (HasPhdrs) {
    if (static_cast<uint64_t>(Phnum) >= std::numeric_limits<uint16_t>::max()) {
      log() << "hotswap: error: growWithTrampolines: program-header count "
            << Phnum << " leaves no room to append a PT_LOAD.\n";
      return nullptr;
    }
    NewPhoff = alignTo(Cursor, static_cast<uint64_t>(alignof(Phdr)));
    std::optional<uint64_t> End = checkedAddUint64(
        NewPhoff, NewPhnum * Phentsize, "relocated phdr table end");
    if (!End)
      return nullptr;
    Cursor = *End;
  }
  if (HasShdrs) {
    if (static_cast<uint64_t>(Shnum) >= std::numeric_limits<uint16_t>::max()) {
      log() << "hotswap: error: growWithTrampolines: section-header count "
            << Shnum << " leaves no room to append the pool section.\n";
      return nullptr;
    }
    NewShoff = alignTo(Cursor, static_cast<uint64_t>(alignof(Shdr)));
    std::optional<uint64_t> End = checkedAddUint64(
        NewShoff, NewShnum * Shentsize, "relocated shdr table end");
    if (!End)
      return nullptr;
    Cursor = *End;
  }
  if (Cursor > std::numeric_limits<size_t>::max()) {
    log()
        << "hotswap: error: growWithTrampolines: grown size exceeds size_t.\n";
    return nullptr;
  }
  const size_t NewSize = static_cast<size_t>(Cursor);

  // getNewMemBuffer zero-initializes, so the alignment gaps between regions are
  // well-defined padding without extra memsets.
  std::unique_ptr<WritableMemoryBuffer> Buf =
      WritableMemoryBuffer::getNewMemBuffer(NewSize);
  if (!Buf) {
    log() << "hotswap: error: growWithTrampolines: "
          << "WritableMemoryBuffer::getNewMemBuffer(" << NewSize
          << ") failed (out of memory).\n";
    return nullptr;
  }

  uint8_t *Out = reinterpret_cast<uint8_t *>(Buf->getBufferStart());
  // 1. Original bytes verbatim -- nothing shifts.
  std::memcpy(Out, Input, InputSize);
  // 2. Trampoline pool at its fresh, page-aligned file offset / vaddr.
  size_t Pos = static_cast<size_t>(PoolFileOff);
  for (const Trampoline &T : Trampolines) {
    std::memcpy(Out + Pos, T.Bytes.data(), T.Bytes.size());
    Pos += T.Bytes.size();
  }
  // 3. Relocated program-header table + appended PT_LOAD for the pool.
  if (HasPhdrs) {
    std::memcpy(Out + NewPhoff, Input + Phoff,
                static_cast<size_t>(Phnum) * Phentsize);
    Phdr PoolPhdr{};
    PoolPhdr.p_type = ELF::PT_LOAD;
    PoolPhdr.p_flags = ELF::PF_R | ELF::PF_X;
    PoolPhdr.p_offset = PoolFileOff;
    PoolPhdr.p_vaddr = PoolVAddr;
    PoolPhdr.p_paddr = PoolVAddr;
    PoolPhdr.p_filesz = TrampTotal;
    PoolPhdr.p_memsz = TrampTotal;
    PoolPhdr.p_align = TrampolinePoolAlign;
    std::memcpy(Out + NewPhoff + static_cast<uint64_t>(Phnum) * Phentsize,
                &PoolPhdr, sizeof(PoolPhdr));
    std::memcpy(Out + offsetof(Ehdr, e_phoff), &NewPhoff, sizeof(NewPhoff));
    uint16_t NewPhnum16 = static_cast<uint16_t>(NewPhnum);
    std::memcpy(Out + offsetof(Ehdr, e_phnum), &NewPhnum16, sizeof(NewPhnum16));
  }
  // 4. Relocated section-header table + appended pool section. The section has
  // an empty name (sh_name == 0): the loader ignores section headers and tools
  // still disassemble it by flags, so no .shstrtab surgery is needed.
  if (HasShdrs) {
    std::memcpy(Out + NewShoff, Input + Shoff,
                static_cast<size_t>(Shnum) * Shentsize);
    Shdr PoolShdr{};
    PoolShdr.sh_name = 0;
    PoolShdr.sh_type = ELF::SHT_PROGBITS;
    PoolShdr.sh_flags = ELF::SHF_ALLOC | ELF::SHF_EXECINSTR;
    PoolShdr.sh_addr = PoolVAddr;
    PoolShdr.sh_offset = PoolFileOff;
    PoolShdr.sh_size = TrampTotal;
    PoolShdr.sh_addralign = TrampolinePoolAlign;
    std::memcpy(Out + NewShoff + static_cast<uint64_t>(Shnum) * Shentsize,
                &PoolShdr, sizeof(PoolShdr));
    std::memcpy(Out + offsetof(Ehdr, e_shoff), &NewShoff, sizeof(NewShoff));
    uint16_t NewShnum16 = static_cast<uint16_t>(NewShnum);
    std::memcpy(Out + offsetof(Ehdr, e_shnum), &NewShnum16, sizeof(NewShnum16));
  }

  log() << "hotswap: growWithTrampolines: appended " << Trampolines.size()
        << (Trampolines.size() == 1 ? " trampoline (" : " trampolines (")
        << TrampTotal << " bytes) at vaddr 0x" << utohexstr(PoolVAddr)
        << " (file 0x" << utohexstr(PoolFileOff) << "); grew ELF from "
        << InputSize << " to " << NewSize << " bytes.\n";
  return Buf;
}

} // namespace hotswap
} // namespace COMGR
