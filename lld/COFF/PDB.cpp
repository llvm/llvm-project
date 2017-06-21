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
#include "Config.h"
#include "Error.h"
#include "SymbolTable.h"
#include "Symbols.h"
#include "llvm/DebugInfo/CodeView/CVDebugRecord.h"
#include "llvm/DebugInfo/CodeView/CVTypeVisitor.h"
#include "llvm/DebugInfo/CodeView/DebugSubsectionRecord.h"
#include "llvm/DebugInfo/CodeView/DebugSubsectionVisitor.h"
#include "llvm/DebugInfo/CodeView/LazyRandomTypeCollection.h"
#include "llvm/DebugInfo/CodeView/SymbolDumper.h"
#include "llvm/DebugInfo/CodeView/TypeDumpVisitor.h"
#include "llvm/DebugInfo/CodeView/TypeStreamMerger.h"
#include "llvm/DebugInfo/CodeView/TypeTableBuilder.h"
#include "llvm/DebugInfo/MSF/MSFBuilder.h"
#include "llvm/DebugInfo/MSF/MSFCommon.h"
#include "llvm/DebugInfo/PDB/Native/DbiStream.h"
#include "llvm/DebugInfo/PDB/Native/DbiStreamBuilder.h"
#include "llvm/DebugInfo/PDB/Native/InfoStream.h"
#include "llvm/DebugInfo/PDB/Native/InfoStreamBuilder.h"
#include "llvm/DebugInfo/PDB/Native/PDBFile.h"
#include "llvm/DebugInfo/PDB/Native/PDBFileBuilder.h"
#include "llvm/DebugInfo/PDB/Native/DbiModuleDescriptorBuilder.h"
#include "llvm/DebugInfo/PDB/Native/PDBStringTableBuilder.h"
#include "llvm/DebugInfo/PDB/Native/PDBTypeServerHandler.h"
#include "llvm/DebugInfo/PDB/Native/TpiStream.h"
#include "llvm/DebugInfo/PDB/Native/TpiStreamBuilder.h"
#include "llvm/Object/COFF.h"
#include "llvm/Support/BinaryByteStream.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/FileOutputBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ScopedPrinter.h"
#include <memory>

using namespace lld;
using namespace lld::coff;
using namespace llvm;
using namespace llvm::codeview;

using llvm::object::coff_section;

static ExitOnError ExitOnErr;

// Returns a list of all SectionChunks.
static void addSectionContribs(SymbolTable *Symtab, pdb::DbiStreamBuilder &DbiBuilder) {
  for (Chunk *C : Symtab->getChunks())
    if (auto *SC = dyn_cast<SectionChunk>(C))
      DbiBuilder.addSectionContrib(SC->File->ModuleDBI, SC->Header);
}

static SectionChunk *findByName(std::vector<SectionChunk *> &Sections,
                                StringRef Name) {
  for (SectionChunk *C : Sections)
    if (C->getSectionName() == Name)
      return C;
  return nullptr;
}

static ArrayRef<uint8_t> consumeDebugMagic(ArrayRef<uint8_t> Data,
                                           StringRef SecName) {
  // First 4 bytes are section magic.
  if (Data.size() < 4)
    fatal(SecName + " too short");
  if (support::endian::read32le(Data.data()) != COFF::DEBUG_SECTION_MAGIC)
    fatal(SecName + " has an invalid magic");
  return Data.slice(4);
}

static ArrayRef<uint8_t> getDebugSection(ObjectFile *File, StringRef SecName) {
  if (SectionChunk *Sec = findByName(File->getDebugChunks(), SecName))
    return consumeDebugMagic(Sec->getContents(), SecName);
  return {};
}

static void addTypeInfo(pdb::TpiStreamBuilder &TpiBuilder,
                        TypeTableBuilder &TypeTable) {
  // Start the TPI or IPI stream header.
  TpiBuilder.setVersionHeader(pdb::PdbTpiV80);

  // Flatten the in memory type table.
  TypeTable.ForEachRecord([&](TypeIndex TI, ArrayRef<uint8_t> Rec) {
    // FIXME: Hash types.
    TpiBuilder.addTypeRecord(Rec, None);
  });
}

static void mergeDebugT(ObjectFile *File,
                        TypeTableBuilder &IDTable,
                        TypeTableBuilder &TypeTable,
                        SmallVectorImpl<TypeIndex> &TypeIndexMap,
                        pdb::PDBTypeServerHandler &Handler) {
  ArrayRef<uint8_t> Data = getDebugSection(File, ".debug$T");
  if (Data.empty())
    return;

  BinaryByteStream Stream(Data, support::little);
  CVTypeArray Types;
  BinaryStreamReader Reader(Stream);
  Handler.addSearchPath(sys::path::parent_path(File->getName()));
  if (auto EC = Reader.readArray(Types, Reader.getLength()))
    fatal(EC, "Reader::readArray failed");
  if (auto Err = mergeTypeAndIdRecords(IDTable, TypeTable,
                                                 TypeIndexMap, &Handler, Types))
    fatal(Err, "codeview::mergeTypeStreams failed");
}

// Allocate memory for a .debug$S section and relocate it.
static ArrayRef<uint8_t> relocateDebugChunk(BumpPtrAllocator &Alloc,
                                            SectionChunk *DebugChunk) {
  uint8_t *Buffer = Alloc.Allocate<uint8_t>(DebugChunk->getSize());
  assert(DebugChunk->OutputSectionOff == 0 &&
         "debug sections should not be in output sections");
  DebugChunk->writeTo(Buffer);
  return consumeDebugMagic(makeArrayRef(Buffer, DebugChunk->getSize()),
                           ".debug$S");
}

// Add all object files to the PDB. Merge .debug$T sections into IpiData and
// TpiData.
static void addObjectsToPDB(BumpPtrAllocator &Alloc, SymbolTable *Symtab,
                            pdb::PDBFileBuilder &Builder,
                            TypeTableBuilder &TypeTable,
                            TypeTableBuilder &IDTable) {
  // Follow type servers.  If the same type server is encountered more than
  // once for this instance of `PDBTypeServerHandler` (for example if many
  // object files reference the same TypeServer), the types from the
  // TypeServer will only be visited once.
  pdb::PDBTypeServerHandler Handler;

  // PDBs use a single global string table for filenames in the file checksum
  // table.
  auto PDBStrTab = std::make_shared<DebugStringTableSubsection>();

  // Visit all .debug$T sections to add them to Builder.
  for (ObjectFile *File : Symtab->ObjectFiles) {
    // Add a module descriptor for every object file. We need to put an absolute
    // path to the object into the PDB. If this is a plain object, we make its
    // path absolute. If it's an object in an archive, we make the archive path
    // absolute.
    bool InArchive = !File->ParentName.empty();
    SmallString<128> Path = InArchive ? File->ParentName : File->getName();
    sys::fs::make_absolute(Path);
    StringRef Name = InArchive ? File->getName() : StringRef(Path);
    File->ModuleDBI = &ExitOnErr(Builder.getDbiBuilder().addModuleInfo(Name));
    File->ModuleDBI->setObjFileName(Path);

    // Before we can process symbol substreams from .debug$S, we need to process
    // type information, file checksums, and the string table.  Add type info to
    // the PDB first, so that we can get the map from object file type and item
    // indices to PDB type and item indices.
    SmallVector<TypeIndex, 128> TypeIndexMap;
    mergeDebugT(File, IDTable, TypeTable, TypeIndexMap, Handler);

    // Now do all line info.
    for (SectionChunk *DebugChunk : File->getDebugChunks()) {
      if (!DebugChunk->isLive() || DebugChunk->getSectionName() != ".debug$S")
        continue;

      ArrayRef<uint8_t> RelocatedDebugContents =
          relocateDebugChunk(Alloc, DebugChunk);
      if (RelocatedDebugContents.empty())
        continue;

      DebugSubsectionArray Subsections;
      BinaryStreamReader Reader(RelocatedDebugContents, support::little);
      ExitOnErr(Reader.readArray(Subsections, RelocatedDebugContents.size()));

      DebugStringTableSubsectionRef CVStrTab;
      DebugChecksumsSubsectionRef Checksums;
      for (const DebugSubsectionRecord &SS : Subsections) {
        switch (SS.kind()) {
        case DebugSubsectionKind::StringTable:
          ExitOnErr(CVStrTab.initialize(SS.getRecordData()));
          break;
        case DebugSubsectionKind::FileChecksums:
          ExitOnErr(Checksums.initialize(SS.getRecordData()));
          break;
        case DebugSubsectionKind::Lines:
          // We can add the relocated line table directly to the PDB without
          // modification because the file checksum offsets will stay the same.
          File->ModuleDBI->addDebugSubsection(SS);
          break;
        default:
          // FIXME: Process the rest of the subsections.
          break;
        }
      }

      if (Checksums.valid()) {
        // Make a new file checksum table that refers to offsets in the PDB-wide
        // string table. Generally the string table subsection appears after the
        // checksum table, so we have to do this after looping over all the
        // subsections.
        if (!CVStrTab.valid())
          fatal(".debug$S sections must have both a string table subsection "
                "and a checksum subsection table or neither");
        auto NewChecksums =
            make_unique<DebugChecksumsSubsection>(*PDBStrTab);
        for (FileChecksumEntry &FC : Checksums) {
          StringRef FileName = ExitOnErr(CVStrTab.getString(FC.FileNameOffset));
          ExitOnErr(Builder.getDbiBuilder().addModuleSourceFile(
              *File->ModuleDBI, FileName));
          NewChecksums->addChecksum(FileName, FC.Kind, FC.Checksum);
        }
        File->ModuleDBI->addDebugSubsection(std::move(NewChecksums));
      }
    }
  }

  Builder.getStringTableBuilder().setStrings(*PDBStrTab);

  // Construct TPI stream contents.
  addTypeInfo(Builder.getTpiBuilder(), TypeTable);

  // Construct IPI stream contents.
  addTypeInfo(Builder.getIpiBuilder(), IDTable);
}

// Creates a PDB file.
void coff::createPDB(StringRef Path, SymbolTable *Symtab,
                     ArrayRef<uint8_t> SectionTable,
                     const llvm::codeview::DebugInfo *DI) {
  BumpPtrAllocator Alloc;
  pdb::PDBFileBuilder Builder(Alloc);
  ExitOnErr(Builder.initialize(4096)); // 4096 is blocksize

  // Create streams in MSF for predefined streams, namely
  // PDB, TPI, DBI and IPI.
  for (int I = 0; I < (int)pdb::kSpecialStreamCount; ++I)
    ExitOnErr(Builder.getMsfBuilder().addStream(0));

  // Add an Info stream.
  auto &InfoBuilder = Builder.getInfoBuilder();
  InfoBuilder.setAge(DI ? DI->PDB70.Age : 0);

  pdb::PDB_UniqueId uuid{};
  if (DI)
    memcpy(&uuid, &DI->PDB70.Signature, sizeof(uuid));
  InfoBuilder.setGuid(uuid);
  // Should be the current time, but set 0 for reproducibilty.
  InfoBuilder.setSignature(0);
  InfoBuilder.setVersion(pdb::PdbRaw_ImplVer::PdbImplVC70);

  // Add an empty DPI stream.
  pdb::DbiStreamBuilder &DbiBuilder = Builder.getDbiBuilder();
  DbiBuilder.setVersionHeader(pdb::PdbDbiV110);

  TypeTableBuilder TypeTable(BAlloc);
  TypeTableBuilder IDTable(BAlloc);
  addObjectsToPDB(Alloc, Symtab, Builder, TypeTable, IDTable);

  // Add Section Contributions.
  addSectionContribs(Symtab, DbiBuilder);

  // Add Section Map stream.
  ArrayRef<object::coff_section> Sections = {
      (const object::coff_section *)SectionTable.data(),
      SectionTable.size() / sizeof(object::coff_section)};
  std::vector<pdb::SecMapEntry> SectionMap =
      pdb::DbiStreamBuilder::createSectionMap(Sections);
  DbiBuilder.setSectionMap(SectionMap);

  ExitOnErr(DbiBuilder.addModuleInfo("* Linker *"));

  // Add COFF section header stream.
  ExitOnErr(
      DbiBuilder.addDbgStream(pdb::DbgHeaderType::SectionHdr, SectionTable));

  // Write to a file.
  ExitOnErr(Builder.commit(Path));
}
