//===- bolt/Rewrite/ExecutableFileMemoryManager.cpp -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bolt/Rewrite/ExecutableFileMemoryManager.h"
#include "bolt/Rewrite/RewriteInstance.h"
#include "llvm/Support/MemAlloc.h"

#undef  DEBUG_TYPE
#define DEBUG_TYPE "efmm"

using namespace llvm;
using namespace object;
using namespace bolt;

namespace llvm {

namespace bolt {

uint8_t *ExecutableFileMemoryManager::allocateSection(
    uintptr_t Size, unsigned Alignment, unsigned SectionID,
    StringRef SectionName, bool IsCode, bool IsReadOnly) {
  uint8_t *Ret = static_cast<uint8_t *>(llvm::allocate_buffer(Size, Alignment));
  AllocatedSections.push_back(AllocInfo{Ret, Size, Alignment});

  // Register a debug section as a note section.
  if (!ObjectsLoaded && RewriteInstance::isDebugSection(SectionName)) {
    BinarySection &Section =
        BC.registerOrUpdateNoteSection(SectionName, Ret, Size, Alignment);
    Section.setSectionID(SectionID);
    assert(!Section.isAllocatable() && "note sections cannot be allocatable");
    return Ret;
  }

  if (!IsCode && (SectionName == ".strtab" || SectionName == ".symtab" ||
                  SectionName == "" || SectionName.startswith(".rela.")))
    return Ret;

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
    Section->updateContents(Ret, Size);
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
        BinarySection::getFlags(IsReadOnly, IsCode, true), Ret, Size,
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
           << Ret << ", ID = " << SectionID << "\n";
  });

  Section->setSectionID(SectionID);

  return Ret;
}

ExecutableFileMemoryManager::~ExecutableFileMemoryManager() {
  for (const AllocInfo &AI : AllocatedSections)
    llvm::deallocate_buffer(AI.Address, AI.Size, AI.Alignment);
}

} // namespace bolt

} // namespace llvm
