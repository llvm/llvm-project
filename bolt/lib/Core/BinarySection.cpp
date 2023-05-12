//===- bolt/Core/BinarySection.cpp - Section in a binary file -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the BinarySection class.
//
//===----------------------------------------------------------------------===//

#include "bolt/Core/BinarySection.h"
#include "bolt/Core/BinaryContext.h"
#include "bolt/Utils/Utils.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Support/CommandLine.h"

#define DEBUG_TYPE "bolt"

using namespace llvm;
using namespace bolt;

namespace opts {
extern cl::opt<bool> PrintRelocations;
extern cl::opt<bool> HotData;
} // namespace opts

uint64_t BinarySection::Count = 0;

bool BinarySection::isELF() const { return BC.isELF(); }

bool BinarySection::isMachO() const { return BC.isMachO(); }

uint64_t
BinarySection::hash(const BinaryData &BD,
                    std::map<const BinaryData *, uint64_t> &Cache) const {
  auto Itr = Cache.find(&BD);
  if (Itr != Cache.end())
    return Itr->second;

  hash_code Hash =
      hash_combine(hash_value(BD.getSize()), hash_value(BD.getSectionName()));

  Cache[&BD] = Hash;

  if (!containsRange(BD.getAddress(), BD.getSize()))
    return Hash;

  uint64_t Offset = BD.getAddress() - getAddress();
  const uint64_t EndOffset = BD.getEndAddress() - getAddress();
  auto Begin = Relocations.lower_bound(Relocation{Offset, 0, 0, 0, 0});
  auto End = Relocations.upper_bound(Relocation{EndOffset, 0, 0, 0, 0});
  const StringRef Contents = getContents();

  while (Begin != End) {
    const Relocation &Rel = *Begin++;
    Hash = hash_combine(
        Hash, hash_value(Contents.substr(Offset, Begin->Offset - Offset)));
    if (BinaryData *RelBD = BC.getBinaryDataByName(Rel.Symbol->getName()))
      Hash = hash_combine(Hash, hash(*RelBD, Cache));
    Offset = Rel.Offset + Rel.getSize();
  }

  Hash = hash_combine(Hash,
                      hash_value(Contents.substr(Offset, EndOffset - Offset)));

  Cache[&BD] = Hash;

  return Hash;
}

void BinarySection::emitAsData(MCStreamer &Streamer,
                               const Twine &SectionName) const {
  StringRef SectionContents = getContents();
  MCSectionELF *ELFSection =
      BC.Ctx->getELFSection(SectionName, getELFType(), getELFFlags());

  Streamer.switchSection(ELFSection);
  Streamer.emitValueToAlignment(getAlign());

  if (BC.HasRelocations && opts::HotData && isReordered())
    Streamer.emitLabel(BC.Ctx->getOrCreateSymbol("__hot_data_start"));

  LLVM_DEBUG(dbgs() << "BOLT-DEBUG: emitting "
                    << (isAllocatable() ? "" : "non-")
                    << "allocatable data section " << SectionName << '\n');

  if (!hasRelocations()) {
    Streamer.emitBytes(SectionContents);
  } else {
    uint64_t SectionOffset = 0;
    for (const Relocation &Relocation : relocations()) {
      assert(Relocation.Offset < SectionContents.size() && "overflow detected");
      // Skip undefined symbols.
      if (BC.UndefinedSymbols.count(Relocation.Symbol))
        continue;
      if (SectionOffset < Relocation.Offset) {
        Streamer.emitBytes(SectionContents.substr(
            SectionOffset, Relocation.Offset - SectionOffset));
        SectionOffset = Relocation.Offset;
      }
      LLVM_DEBUG(dbgs() << "BOLT-DEBUG: emitting relocation for symbol "
                        << (Relocation.Symbol ? Relocation.Symbol->getName()
                                              : StringRef("<none>"))
                        << " at offset 0x"
                        << Twine::utohexstr(Relocation.Offset) << " with size "
                        << Relocation::getSizeForType(Relocation.Type) << '\n');
      size_t RelocationSize = Relocation.emit(&Streamer);
      SectionOffset += RelocationSize;
    }
    assert(SectionOffset <= SectionContents.size() && "overflow error");
    if (SectionOffset < SectionContents.size())
      Streamer.emitBytes(SectionContents.substr(SectionOffset));
  }

  if (BC.HasRelocations && opts::HotData && isReordered())
    Streamer.emitLabel(BC.Ctx->getOrCreateSymbol("__hot_data_end"));
}

void BinarySection::flushPendingRelocations(raw_pwrite_stream &OS,
                                            SymbolResolverFuncTy Resolver) {
  if (PendingRelocations.empty() && Patches.empty())
    return;

  const uint64_t SectionAddress = getAddress();

  // We apply relocations to original section contents. For allocatable sections
  // this means using their input file offsets, since the output file offset
  // could change (e.g. for new instance of .text). For non-allocatable
  // sections, the output offset should always be a valid one.
  const uint64_t SectionFileOffset =
      isAllocatable() ? getInputFileOffset() : getOutputFileOffset();
  LLVM_DEBUG(
      dbgs() << "BOLT-DEBUG: flushing pending relocations for section "
             << getName() << '\n'
             << "  address: 0x" << Twine::utohexstr(SectionAddress) << '\n'
             << "  offset: 0x" << Twine::utohexstr(SectionFileOffset) << '\n');

  for (BinaryPatch &Patch : Patches)
    OS.pwrite(Patch.Bytes.data(), Patch.Bytes.size(),
              SectionFileOffset + Patch.Offset);

  for (Relocation &Reloc : PendingRelocations) {
    uint64_t Value = Reloc.Addend;
    if (Reloc.Symbol)
      Value += Resolver(Reloc.Symbol);

    Value = Relocation::encodeValue(Reloc.Type, Value,
                                    SectionAddress + Reloc.Offset);

    OS.pwrite(reinterpret_cast<const char *>(&Value),
              Relocation::getSizeForType(Reloc.Type),
              SectionFileOffset + Reloc.Offset);

    LLVM_DEBUG(
        dbgs() << "BOLT-DEBUG: writing value 0x" << Twine::utohexstr(Value)
               << " of size " << Relocation::getSizeForType(Reloc.Type)
               << " at section offset 0x" << Twine::utohexstr(Reloc.Offset)
               << " address 0x"
               << Twine::utohexstr(SectionAddress + Reloc.Offset)
               << " file offset 0x"
               << Twine::utohexstr(SectionFileOffset + Reloc.Offset) << '\n';);
  }

  clearList(PendingRelocations);
}

BinarySection::~BinarySection() {
  if (isReordered()) {
    delete[] getData();
    return;
  }

  if (!isAllocatable() && !hasValidSectionID() &&
      (!hasSectionRef() ||
       OutputContents.data() != getContents(Section).data())) {
    delete[] getOutputData();
  }
}

void BinarySection::clearRelocations() { clearList(Relocations); }

void BinarySection::print(raw_ostream &OS) const {
  OS << getName() << ", "
     << "0x" << Twine::utohexstr(getAddress()) << ", " << getSize() << " (0x"
     << Twine::utohexstr(getOutputAddress()) << ", " << getOutputSize() << ")"
     << ", data = " << getData() << ", output data = " << getOutputData();

  if (isAllocatable())
    OS << " (allocatable)";

  if (isVirtual())
    OS << " (virtual)";

  if (isTLS())
    OS << " (tls)";

  if (opts::PrintRelocations)
    for (const Relocation &R : relocations())
      OS << "\n  " << R;
}

BinarySection::RelocationSetType
BinarySection::reorderRelocations(bool Inplace) const {
  assert(PendingRelocations.empty() &&
         "reodering pending relocations not supported");
  RelocationSetType NewRelocations;
  for (const Relocation &Rel : relocations()) {
    uint64_t RelAddr = Rel.Offset + getAddress();
    BinaryData *BD = BC.getBinaryDataContainingAddress(RelAddr);
    BD = BD->getAtomicRoot();
    assert(BD);

    if ((!BD->isMoved() && !Inplace) || BD->isJumpTable())
      continue;

    Relocation NewRel(Rel);
    uint64_t RelOffset = RelAddr - BD->getAddress();
    NewRel.Offset = BD->getOutputOffset() + RelOffset;
    assert(NewRel.Offset < getSize());
    LLVM_DEBUG(dbgs() << "BOLT-DEBUG: moving " << Rel << " -> " << NewRel
                      << "\n");
    auto Res = NewRelocations.emplace(std::move(NewRel));
    (void)Res;
    assert(Res.second && "Can't overwrite existing relocation");
  }
  return NewRelocations;
}

void BinarySection::reorderContents(const std::vector<BinaryData *> &Order,
                                    bool Inplace) {
  IsReordered = true;

  Relocations = reorderRelocations(Inplace);

  std::string Str;
  raw_string_ostream OS(Str);
  const char *Src = Contents.data();
  LLVM_DEBUG(dbgs() << "BOLT-DEBUG: reorderContents for " << Name << "\n");
  for (BinaryData *BD : Order) {
    assert((BD->isMoved() || !Inplace) && !BD->isJumpTable());
    assert(BD->isAtomic() && BD->isMoveable());
    const uint64_t SrcOffset = BD->getAddress() - getAddress();
    assert(SrcOffset < Contents.size());
    assert(SrcOffset == BD->getOffset());
    while (OS.tell() < BD->getOutputOffset())
      OS.write((unsigned char)0);
    LLVM_DEBUG(dbgs() << "BOLT-DEBUG: " << BD->getName() << " @ " << OS.tell()
                      << "\n");
    OS.write(&Src[SrcOffset], BD->getOutputSize());
  }
  if (Relocations.empty()) {
    // If there are no existing relocations, tack a phony one at the end
    // of the reordered segment to force LLVM to recognize and map this
    // section.
    MCSymbol *ZeroSym = BC.registerNameAtAddress("Zero", 0, 0, 0);
    addRelocation(OS.tell(), ZeroSym, Relocation::getAbs64(), 0xdeadbeef);

    uint64_t Zero = 0;
    OS.write(reinterpret_cast<const char *>(&Zero), sizeof(Zero));
  }
  auto *NewData = reinterpret_cast<char *>(copyByteArray(OS.str()));
  Contents = OutputContents = StringRef(NewData, OS.str().size());
  OutputSize = Contents.size();
}

std::string BinarySection::encodeELFNote(StringRef NameStr, StringRef DescStr,
                                         uint32_t Type) {
  std::string Str;
  raw_string_ostream OS(Str);
  const uint32_t NameSz = NameStr.size() + 1;
  const uint32_t DescSz = DescStr.size();
  OS.write(reinterpret_cast<const char *>(&(NameSz)), 4);
  OS.write(reinterpret_cast<const char *>(&(DescSz)), 4);
  OS.write(reinterpret_cast<const char *>(&(Type)), 4);
  OS << NameStr << '\0';
  for (uint64_t I = NameSz; I < alignTo(NameSz, 4); ++I)
    OS << '\0';
  OS << DescStr;
  for (uint64_t I = DescStr.size(); I < alignTo(DescStr.size(), 4); ++I)
    OS << '\0';
  return OS.str();
}
