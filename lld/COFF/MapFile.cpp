//===- MapFile.cpp --------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the /lldmap option. It shows lists in order and
// hierarchically the output sections, input sections, input files and
// symbol:
//
// Address  Size     Align Out     In      File    Symbol
// =================================================================
// 00201000 00000015     4 .text
// 00201000 0000000e     4         .text
// 00201000 0000000e     4                 test.o
// 0020100e 00000000     0                         local
// 00201005 00000000     0                         f(int)
//
//===----------------------------------------------------------------------===//

#include "MapFile.h"
#include "Error.h"
#include "Symbols.h"
#include "Writer.h"

#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::object;

using namespace lld;
using namespace lld::coff;

static void writeOutSecLine(raw_fd_ostream &OS, uint64_t Address, uint64_t Size,
                            uint64_t Align, StringRef Name) {
  OS << format("%08llx %08llx %5lld ", Address, Size, Align)
     << left_justify(Name, 7);
}

static void writeInSecLine(raw_fd_ostream &OS, uint64_t Address, uint64_t Size,
                           uint64_t Align, StringRef Name) {
  // Pass an empty name to align the text to the correct column.
  writeOutSecLine(OS, Address, Size, Align, "");
  OS << ' ' << left_justify(Name, 7);
}

static void writeFileLine(raw_fd_ostream &OS, uint64_t Address, uint64_t Size,
                          uint64_t Align, StringRef Name) {
  // Pass an empty name to align the text to the correct column.
  writeInSecLine(OS, Address, Size, Align, "");
  OS << ' ' << left_justify(Name, 7);
}

static void writeSymbolLine(raw_fd_ostream &OS, uint64_t Address, uint64_t Size,
                            StringRef Name) {
  // Pass an empty name to align the text to the correct column.
  writeFileLine(OS, Address, Size, 0, "");
  OS << ' ' << left_justify(Name, 7);
}

static void writeSectionChunk(raw_fd_ostream &OS, const SectionChunk *SC,
                              StringRef &PrevName) {
  StringRef Name = SC->getSectionName();
  if (Name != PrevName) {
    writeInSecLine(OS, SC->getRVA(), SC->getSize(), SC->getAlign(), Name);
    OS << '\n';
    PrevName = Name;
  }
  coff::ObjectFile *File = SC->File;
  if (!File)
    return;
  writeFileLine(OS, SC->getRVA(), SC->getSize(), SC->getAlign(),
                toString(File));
  OS << '\n';
  ArrayRef<SymbolBody *> Syms = File->getSymbols();
  for (SymbolBody *Sym : Syms) {
    auto *DR = dyn_cast<DefinedRegular>(Sym);
    if (!DR || DR->getChunk() != SC ||
        DR->getCOFFSymbol().isSectionDefinition())
      continue;
    writeSymbolLine(OS, DR->getRVA(), SC->getSize(), toString(*Sym));
    OS << '\n';
  }
}

static void writeMapFile2(raw_fd_ostream &OS,
                          ArrayRef<OutputSection *> OutputSections) {
  OS << "Address  Size     Align Out     In      File    Symbol\n";

  for (OutputSection *Sec : OutputSections) {
    uint32_t VA = Sec->getRVA();
    writeOutSecLine(OS, VA, Sec->getVirtualSize(), /*Align=*/PageSize,
                    Sec->getName());
    OS << '\n';
    StringRef PrevName = "";
    for (Chunk *C : Sec->getChunks())
      if (const auto *SC = dyn_cast<SectionChunk>(C))
        writeSectionChunk(OS, SC, PrevName);
  }
}

void coff::writeMapFile(ArrayRef<OutputSection *> OutputSections) {
  if (Config->MapFile.empty())
    return;

  std::error_code EC;
  raw_fd_ostream OS(Config->MapFile, EC, sys::fs::F_None);
  if (EC)
    fatal("cannot open " + Config->MapFile + ": " + EC.message());
  writeMapFile2(OS, OutputSections);
}
