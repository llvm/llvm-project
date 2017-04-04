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
#include "llvm/DebugInfo/CodeView/CVTypeDumper.h"
#include "llvm/DebugInfo/CodeView/SymbolDumper.h"
#include "llvm/DebugInfo/CodeView/TypeDatabase.h"
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
#include "llvm/DebugInfo/PDB/Native/PDBTypeServerHandler.h"
#include "llvm/DebugInfo/PDB/Native/StringTableBuilder.h"
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

static SectionChunk *findByName(std::vector<SectionChunk *> &Sections,
                                StringRef Name) {
  for (SectionChunk *C : Sections)
    if (C->getSectionName() == Name)
      return C;
  return nullptr;
}

static ArrayRef<uint8_t> getDebugSection(ObjectFile *File, StringRef SecName) {
  SectionChunk *Sec = findByName(File->getDebugChunks(), SecName);
  if (!Sec)
    return {};

  // First 4 bytes are section magic.
  ArrayRef<uint8_t> Data = Sec->getContents();
  if (Data.size() < 4)
    fatal(SecName + " too short");
  if (read32le(Data.data()) != COFF::DEBUG_SECTION_MAGIC)
    fatal(SecName + " has an invalid magic");
  return Data.slice(4);
}

static void addTypeInfo(pdb::TpiStreamBuilder &TpiBuilder,
                        codeview::TypeTableBuilder &TypeTable) {
  // Start the TPI or IPI stream header.
  TpiBuilder.setVersionHeader(pdb::PdbTpiV80);

  // Flatten the in memory type table.
  TypeTable.ForEachRecord([&](TypeIndex TI, ArrayRef<uint8_t> Rec) {
    // FIXME: Hash types.
    TpiBuilder.addTypeRecord(Rec, None);
  });
}

// Merge .debug$T sections into IpiData and TpiData.
static void mergeDebugT(SymbolTable *Symtab, pdb::PDBFileBuilder &Builder,
                        codeview::TypeTableBuilder &TypeTable,
                        codeview::TypeTableBuilder &IDTable) {
  // Visit all .debug$T sections to add them to Builder.
  for (ObjectFile *File : Symtab->ObjectFiles) {
    ArrayRef<uint8_t> Data = getDebugSection(File, ".debug$T");
    if (Data.empty())
      continue;

    BinaryByteStream Stream(Data, support::little);
    codeview::CVTypeArray Types;
    BinaryStreamReader Reader(Stream);
    // Follow type servers.  If the same type server is encountered more than
    // once for this instance of `PDBTypeServerHandler` (for example if many
    // object files reference the same TypeServer), the types from the
    // TypeServer will only be visited once.
    pdb::PDBTypeServerHandler Handler;
    Handler.addSearchPath(llvm::sys::path::parent_path(File->getName()));
    if (auto EC = Reader.readArray(Types, Reader.getLength()))
      fatal(EC, "Reader::readArray failed");
    if (auto Err =
            codeview::mergeTypeStreams(IDTable, TypeTable, &Handler, Types))
      fatal(Err, "codeview::mergeTypeStreams failed");
  }

  // Construct TPI stream contents.
  addTypeInfo(Builder.getTpiBuilder(), TypeTable);

  // Construct IPI stream contents.
  addTypeInfo(Builder.getIpiBuilder(), IDTable);
}

static void dumpDebugT(ScopedPrinter &W, ObjectFile *File) {
  ListScope LS(W, "DebugT");
  ArrayRef<uint8_t> Data = getDebugSection(File, ".debug$T");
  if (Data.empty())
    return;

  TypeDatabase TDB;
  TypeDumpVisitor TDV(TDB, &W, false);
  // Use a default implementation that does not follow type servers and instead
  // just dumps the contents of the TypeServer2 record.
  CVTypeDumper TypeDumper(TDB);
  if (auto EC = TypeDumper.dump(Data, TDV))
    fatal(EC, "CVTypeDumper::dump failed");
}

static void dumpDebugS(ScopedPrinter &W, ObjectFile *File) {
  ListScope LS(W, "DebugS");
  ArrayRef<uint8_t> Data = getDebugSection(File, ".debug$S");
  if (Data.empty())
    return;

  BinaryByteStream Stream(Data, llvm::support::little);
  CVSymbolArray Symbols;
  BinaryStreamReader Reader(Stream);
  if (auto EC = Reader.readArray(Symbols, Reader.getLength()))
    fatal(EC, "StreamReader.readArray<CVSymbolArray> failed");

  TypeDatabase TDB;
  CVSymbolDumper SymbolDumper(W, TDB, nullptr, false);
  if (auto EC = SymbolDumper.dump(Symbols))
    fatal(EC, "CVSymbolDumper::dump failed");
}

// Dump CodeView debug info. This is for debugging.
static void dumpCodeView(SymbolTable *Symtab) {
  ScopedPrinter W(outs());

  for (ObjectFile *File : Symtab->ObjectFiles) {
    dumpDebugT(W, File);
    dumpDebugS(W, File);
  }
}

// Creates a PDB file.
void coff::createPDB(StringRef Path, SymbolTable *Symtab,
                     ArrayRef<uint8_t> SectionTable,
                     const llvm::codeview::DebugInfo *DI) {
  if (Config->DumpPdb)
    dumpCodeView(Symtab);

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
  auto &DbiBuilder = Builder.getDbiBuilder();
  DbiBuilder.setVersionHeader(pdb::PdbDbiV110);

  codeview::TypeTableBuilder TypeTable(BAlloc);
  codeview::TypeTableBuilder IDTable(BAlloc);
  mergeDebugT(Symtab, Builder, TypeTable, IDTable);

  // Add Section Contributions.
  std::vector<pdb::SectionContrib> Contribs =
      pdb::DbiStreamBuilder::createSectionContribs(getInputSections(Symtab));
  DbiBuilder.setSectionContribs(Contribs);

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
