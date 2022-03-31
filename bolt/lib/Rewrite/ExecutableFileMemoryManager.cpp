//===- bolt/Rewrite/ExecutableFileMemoryManager.cpp -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bolt/Rewrite/ExecutableFileMemoryManager.h"
#include "bolt/Rewrite/RewriteInstance.h"

#undef  DEBUG_TYPE
#define DEBUG_TYPE "efmm"

using namespace llvm;
using namespace object;
using namespace bolt;

namespace llvm {

namespace bolt {

uint8_t *ExecutableFileMemoryManager::allocateSection(intptr_t Size,
                                                      unsigned Alignment,
                                                      unsigned SectionID,
                                                      StringRef SectionName,
                                                      bool IsCode,
                                                      bool IsReadOnly) {
  // Register a debug section as a note section.
  if (!ObjectsLoaded && RewriteInstance::isDebugSection(SectionName)) {
    uint8_t *DataCopy = new uint8_t[Size];
    BinarySection &Section =
        BC.registerOrUpdateNoteSection(SectionName, DataCopy, Size, Alignment);
    Section.setSectionID(SectionID);
    assert(!Section.isAllocatable() && "note sections cannot be allocatable");
    return DataCopy;
  }

  if (!IsCode && (SectionName == ".strtab" || SectionName == ".symtab" ||
                  SectionName == "" || SectionName.startswith(".rela.")))
    return SectionMemoryManager::allocateDataSection(Size, Alignment, SectionID,
                                                     SectionName, IsReadOnly);

  uint8_t *Ret;
  if (IsCode)
    Ret = SectionMemoryManager::allocateCodeSection(Size, Alignment, SectionID,
                                                    SectionName);
  else
    Ret = SectionMemoryManager::allocateDataSection(Size, Alignment, SectionID,
                                                    SectionName, IsReadOnly);

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

  BinarySection &Section = BC.registerOrUpdateSection(
      SectionName, ELF::SHT_PROGBITS,
      BinarySection::getFlags(IsReadOnly, IsCode, true), Ret, Size, Alignment);
  Section.setSectionID(SectionID);
  assert(Section.isAllocatable() &&
         "verify that allocatable is marked as allocatable");

  LLVM_DEBUG(
      dbgs() << "BOLT: allocating "
             << (IsCode ? "code" : (IsReadOnly ? "read-only data" : "data"))
             << " section : " << SectionName << " with size " << Size
             << ", alignment " << Alignment << " at 0x" << Ret
             << ", ID = " << SectionID << "\n");
  return Ret;
}

bool ExecutableFileMemoryManager::finalizeMemory(std::string *ErrMsg) {
  LLVM_DEBUG(dbgs() << "BOLT: finalizeMemory()\n");
  ++ObjectsLoaded;
  return SectionMemoryManager::finalizeMemory(ErrMsg);
}

ExecutableFileMemoryManager::~ExecutableFileMemoryManager() {}

} // namespace bolt

} // namespace llvm
