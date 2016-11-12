//===- PDB.cpp ------------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "PDB.h"
#include "Chunks.h"
#include "Error.h"
#include "SymbolTable.h"
#include "Symbols.h"
#include "llvm/DebugInfo/MSF/MSFBuilder.h"
#include "llvm/DebugInfo/MSF/MSFCommon.h"
#include "llvm/DebugInfo/PDB/Raw/DbiStream.h"
#include "llvm/DebugInfo/PDB/Raw/DbiStreamBuilder.h"
#include "llvm/DebugInfo/PDB/Raw/InfoStream.h"
#include "llvm/DebugInfo/PDB/Raw/InfoStreamBuilder.h"
#include "llvm/DebugInfo/PDB/Raw/PDBFile.h"
#include "llvm/DebugInfo/PDB/Raw/PDBFileBuilder.h"
#include "llvm/DebugInfo/PDB/Raw/TpiStream.h"
#include "llvm/DebugInfo/PDB/Raw/TpiStreamBuilder.h"
#include "llvm/Object/COFF.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/FileOutputBuffer.h"
#include <memory>

using namespace lld;
using namespace lld::coff;
using namespace llvm;
using namespace llvm::support;
using namespace llvm::support::endian;

using llvm::object::coff_section;

static ExitOnError ExitOnErr;

// Returns a list of all SectionChunks.
static std::vector<coff_section> getInputSections(SymbolTable *Symtab) {
  std::vector<coff_section> V;
  for (Chunk *C : Symtab->getChunks())
    if (auto *SC = dyn_cast<SectionChunk>(C))
      V.push_back(*SC->Header);
  return V;
}

// Creates a PDB file.
void coff::createPDB(StringRef Path, SymbolTable *Symtab,
                     ArrayRef<uint8_t> SectionTable) {
  BumpPtrAllocator Alloc;
  pdb::PDBFileBuilder Builder(Alloc);
  ExitOnErr(Builder.initialize(4096)); // 4096 is blocksize

  // Create streams in MSF for predefined streams, namely
  // PDB, TPI, DBI and IPI.
  for (int I = 0; I < (int)pdb::kSpecialStreamCount; ++I)
    ExitOnErr(Builder.getMsfBuilder().addStream(0));

  // Add an Info stream.
  auto &InfoBuilder = Builder.getInfoBuilder();
  InfoBuilder.setAge(1);

  // Should be a random number, 0 for now.
  InfoBuilder.setGuid({});

  // Should be the current time, but set 0 for reproducibilty.
  InfoBuilder.setSignature(0);
  InfoBuilder.setVersion(pdb::PdbRaw_ImplVer::PdbImplVC70);

  // Add an empty DPI stream.
  auto &DbiBuilder = Builder.getDbiBuilder();
  DbiBuilder.setVersionHeader(pdb::PdbDbiV110);

  // Add an empty TPI stream.
  auto &TpiBuilder = Builder.getTpiBuilder();
  TpiBuilder.setVersionHeader(pdb::PdbTpiV80);

  // Add an empty IPI stream.
  auto &IpiBuilder = Builder.getIpiBuilder();
  IpiBuilder.setVersionHeader(pdb::PdbTpiV80);

  // Add Section Contributions.
  ArrayRef<coff_section> InputSections = getInputSections(Symtab);
  std::vector<pdb::SectionContrib> Contribs =
      pdb::DbiStreamBuilder::createSectionContribs(InputSections);
  DbiBuilder.setSectionContribs(Contribs);

  // Add Section Map stream.
  ArrayRef<object::coff_section> Sections = {
      (const object::coff_section *)SectionTable.data(),
      SectionTable.size() / sizeof(object::coff_section)};
  std::vector<pdb::SecMapEntry> SectionMap =
      pdb::DbiStreamBuilder::createSectionMap(Sections);
  DbiBuilder.setSectionMap(SectionMap);

  // Add COFF section header stream.
  ExitOnErr(
      DbiBuilder.addDbgStream(pdb::DbgHeaderType::SectionHdr, SectionTable));

  // Write to a file.
  ExitOnErr(Builder.commit(Path));
}
