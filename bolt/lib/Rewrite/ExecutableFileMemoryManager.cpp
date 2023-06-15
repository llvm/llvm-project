//===- bolt/Rewrite/ExecutableFileMemoryManager.cpp -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bolt/Rewrite/ExecutableFileMemoryManager.h"
#include "bolt/Rewrite/JITLinkLinker.h"
#include "bolt/Rewrite/RewriteInstance.h"
#include "llvm/ExecutionEngine/JITLink/JITLink.h"
#include "llvm/Support/MemAlloc.h"

#undef  DEBUG_TYPE
#define DEBUG_TYPE "efmm"

using namespace llvm;
using namespace object;
using namespace bolt;

namespace llvm {

namespace bolt {

namespace {

SmallVector<jitlink::Section *> orderedSections(jitlink::LinkGraph &G) {
  SmallVector<jitlink::Section *> Sections(
      llvm::map_range(G.sections(), [](auto &S) { return &S; }));
  llvm::sort(Sections, [](const auto *LHS, const auto *RHS) {
    return LHS->getOrdinal() < RHS->getOrdinal();
  });
  return Sections;
}

size_t sectionAlignment(const jitlink::Section &Section) {
  assert(!Section.empty() && "Cannot get alignment for empty section");
  return JITLinkLinker::orderedBlocks(Section).front()->getAlignment();
}

StringRef sectionName(const jitlink::Section &Section,
                      const BinaryContext &BC) {
  auto Name = Section.getName();

  if (BC.isMachO()) {
    // JITLink "normalizes" section names as "SegmentName,SectionName" on
    // Mach-O. BOLT internally refers to sections just by the section name so
    // strip-off the segment name.
    auto SegmentEnd = Name.find(',');
    assert(SegmentEnd != StringRef::npos && "Mach-O segment not found");
    Name = Name.substr(SegmentEnd + 1);
  }

  return Name;
}

struct SectionAllocInfo {
  void *Address;
  size_t Size;
  size_t Alignment;
};

struct AllocInfo {
  SmallVector<SectionAllocInfo, 8> AllocatedSections;

  ~AllocInfo() {
    for (auto &Section : AllocatedSections)
      deallocate_buffer(Section.Address, Section.Size, Section.Alignment);
  }

  SectionAllocInfo allocateSection(const jitlink::Section &Section) {
    auto Size = JITLinkLinker::sectionSize(Section);
    auto Alignment = sectionAlignment(Section);
    auto *Buf = allocate_buffer(Size, Alignment);
    SectionAllocInfo Alloc{Buf, Size, Alignment};
    AllocatedSections.push_back(Alloc);
    return Alloc;
  }
};

struct BOLTInFlightAlloc : ExecutableFileMemoryManager::InFlightAlloc {
  // Even though this is passed using a raw pointer in FinalizedAlloc, we keep
  // it in a unique_ptr as long as possible to enjoy automatic cleanup when
  // something goes wrong.
  std::unique_ptr<AllocInfo> Alloc;

public:
  BOLTInFlightAlloc(std::unique_ptr<AllocInfo> Alloc)
      : Alloc(std::move(Alloc)) {}

  virtual void abandon(OnAbandonedFunction OnAbandoned) override {
    llvm_unreachable("unexpected abandoned allocation");
  }

  virtual void finalize(OnFinalizedFunction OnFinalized) override {
    OnFinalized(ExecutableFileMemoryManager::FinalizedAlloc(
        orc::ExecutorAddr::fromPtr(Alloc.release())));
  }
};

} // anonymous namespace

void ExecutableFileMemoryManager::updateSection(
    const jitlink::Section &JLSection, uint8_t *Contents, size_t Size,
    size_t Alignment) {
  auto SectionID = JLSection.getName();
  auto SectionName = sectionName(JLSection, BC);
  auto Prot = JLSection.getMemProt();
  auto IsCode = (Prot & orc::MemProt::Exec) != orc::MemProt::None;
  auto IsReadOnly = (Prot & orc::MemProt::Write) == orc::MemProt::None;

  // Register a debug section as a note section.
  if (!ObjectsLoaded && RewriteInstance::isDebugSection(SectionName)) {
    BinarySection &Section =
        BC.registerOrUpdateNoteSection(SectionName, Contents, Size, Alignment);
    Section.setSectionID(SectionID);
    assert(!Section.isAllocatable() && "note sections cannot be allocatable");
    return;
  }

  if (!IsCode && (SectionName == ".strtab" || SectionName == ".symtab" ||
                  SectionName == "" || SectionName.startswith(".rela.")))
    return;

  SmallVector<char, 256> Buf;
  if (ObjectsLoaded > 0) {
    if (BC.isELF()) {
      SectionName = (Twine(SectionName) + ".bolt.extra." + Twine(ObjectsLoaded))
                        .toStringRef(Buf);
    } else if (BC.isMachO()) {
      assert((SectionName == "__text" || SectionName == "__data" ||
              SectionName == "__fini" || SectionName == "__setup" ||
              SectionName == "__cstring" || SectionName == "__literal16") &&
             "Unexpected section in the instrumentation library");
      // Sections coming from the instrumentation runtime are prefixed with "I".
      SectionName = ("I" + Twine(SectionName)).toStringRef(Buf);
    }
  }

  BinarySection *Section = nullptr;
  if (!OrgSecPrefix.empty() && SectionName.startswith(OrgSecPrefix)) {
    // Update the original section contents.
    ErrorOr<BinarySection &> OrgSection =
        BC.getUniqueSectionByName(SectionName.substr(OrgSecPrefix.length()));
    assert(OrgSection && OrgSection->isAllocatable() &&
           "Original section must exist and be allocatable.");

    Section = &OrgSection.get();
    Section->updateContents(Contents, Size);
  } else {
    // If the input contains a section with the section name, rename it in the
    // output file to avoid the section name conflict and emit the new section
    // under a unique internal name.
    ErrorOr<BinarySection &> OrgSection =
        BC.getUniqueSectionByName(SectionName);
    bool UsePrefix = false;
    if (OrgSection && OrgSection->hasSectionRef()) {
      OrgSection->setOutputName(OrgSecPrefix + SectionName);
      UsePrefix = true;
    }

    // Register the new section under a unique name to avoid name collision with
    // sections in the input file.
    BinarySection &NewSection = BC.registerOrUpdateSection(
        UsePrefix ? NewSecPrefix + SectionName : SectionName, ELF::SHT_PROGBITS,
        BinarySection::getFlags(IsReadOnly, IsCode, true), Contents, Size,
        Alignment);
    if (UsePrefix)
      NewSection.setOutputName(SectionName);
    Section = &NewSection;
  }

  LLVM_DEBUG({
    dbgs() << "BOLT: allocating "
           << (IsCode ? "code" : (IsReadOnly ? "read-only data" : "data"))
           << " section : " << Section->getOutputName() << " ("
           << Section->getName() << ")"
           << " with size " << Size << ", alignment " << Alignment << " at "
           << Contents << ", ID = " << SectionID << "\n";
  });

  Section->setSectionID(SectionID);
}

void ExecutableFileMemoryManager::allocate(const jitlink::JITLinkDylib *JD,
                                           jitlink::LinkGraph &G,
                                           OnAllocatedFunction OnAllocated) {
  auto Alloc = std::make_unique<AllocInfo>();

  for (auto *Section : orderedSections(G)) {
    if (Section->empty())
      continue;

    auto SectionAlloc = Alloc->allocateSection(*Section);
    updateSection(*Section, static_cast<uint8_t *>(SectionAlloc.Address),
                  SectionAlloc.Size, SectionAlloc.Alignment);

    size_t CurrentOffset = 0;
    auto *Buf = static_cast<char *>(SectionAlloc.Address);
    for (auto *Block : JITLinkLinker::orderedBlocks(*Section)) {
      CurrentOffset = jitlink::alignToBlock(CurrentOffset, *Block);
      auto BlockSize = Block->getSize();
      auto *BlockBuf = Buf + CurrentOffset;

      if (Block->isZeroFill())
        std::memset(BlockBuf, 0, BlockSize);
      else
        std::memcpy(BlockBuf, Block->getContent().data(), BlockSize);

      Block->setMutableContent({BlockBuf, Block->getSize()});
      CurrentOffset += BlockSize;
    }
  }

  OnAllocated(std::make_unique<BOLTInFlightAlloc>(std::move(Alloc)));
}

void ExecutableFileMemoryManager::deallocate(
    std::vector<FinalizedAlloc> Allocs, OnDeallocatedFunction OnDeallocated) {
  for (auto &Alloc : Allocs)
    delete Alloc.release().toPtr<AllocInfo *>();

  OnDeallocated(Error::success());
}

} // namespace bolt

} // namespace llvm
