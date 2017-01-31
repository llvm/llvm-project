//===- MapFile.cpp --------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the -Map option. It shows lists in order and
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
#include "InputFiles.h"
#include "Strings.h"

#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::object;

using namespace lld;
using namespace lld::elf;

static void writeOutSecLine(raw_fd_ostream &OS, int Width, uint64_t Address,
                            uint64_t Size, uint64_t Align, StringRef Name) {
  OS << format("%0*llx %0*llx %5lld ", Width, Address, Width, Size, Align)
     << left_justify(Name, 7);
}

static void writeInSecLine(raw_fd_ostream &OS, int Width, uint64_t Address,
                           uint64_t Size, uint64_t Align, StringRef Name) {
  // Pass an empty name to align the text to the correct column.
  writeOutSecLine(OS, Width, Address, Size, Align, "");
  OS << ' ' << left_justify(Name, 7);
}

static void writeFileLine(raw_fd_ostream &OS, int Width, uint64_t Address,
                          uint64_t Size, uint64_t Align, StringRef Name) {
  // Pass an empty name to align the text to the correct column.
  writeInSecLine(OS, Width, Address, Size, Align, "");
  OS << ' ' << left_justify(Name, 7);
}

static void writeSymbolLine(raw_fd_ostream &OS, int Width, uint64_t Address,
                            uint64_t Size, StringRef Name) {
  // Pass an empty name to align the text to the correct column.
  writeFileLine(OS, Width, Address, Size, 0, "");
  OS << ' ' << left_justify(Name, 7);
}

template <class ELFT>
static void writeInputSection(raw_fd_ostream &OS, const InputSection<ELFT> *IS,
                              StringRef &PrevName) {
  int Width = ELFT::Is64Bits ? 16 : 8;
  StringRef Name = IS->Name;
  if (Name != PrevName) {
    writeInSecLine(OS, Width, IS->OutSec->Addr + IS->OutSecOff, IS->getSize(),
                   IS->Alignment, Name);
    OS << '\n';
    PrevName = Name;
  }

  elf::ObjectFile<ELFT> *File = IS->getFile();
  if (!File)
    return;
  writeFileLine(OS, Width, IS->OutSec->Addr + IS->OutSecOff, IS->getSize(),
                IS->Alignment, toString(File));
  OS << '\n';

  for (SymbolBody *Sym : File->getSymbols()) {
    auto *DR = dyn_cast<DefinedRegular<ELFT>>(Sym);
    if (!DR)
      continue;
    if (DR->Section != IS)
      continue;
    if (DR->isSection())
      continue;
    writeSymbolLine(OS, Width, Sym->getVA<ELFT>(), Sym->getSize<ELFT>(),
                    toString(*Sym));
    OS << '\n';
  }
}

template <class ELFT>
static void writeMapFile2(raw_fd_ostream &OS,
                          ArrayRef<OutputSectionBase *> OutputSections) {
  int Width = ELFT::Is64Bits ? 16 : 8;

  OS << left_justify("Address", Width) << ' ' << left_justify("Size", Width)
     << " Align Out     In      File    Symbol\n";

  for (OutputSectionBase *Sec : OutputSections) {
    writeOutSecLine(OS, Width, Sec->Addr, Sec->Size, Sec->Addralign,
                    Sec->getName());
    OS << '\n';

    StringRef PrevName = "";
    Sec->forEachInputSection([&](InputSectionData *S) {
      if (const auto *IS = dyn_cast<InputSection<ELFT>>(S))
        writeInputSection(OS, IS, PrevName);
    });
  }
}

template <class ELFT>
void elf::writeMapFile(ArrayRef<OutputSectionBase *> OutputSections) {
  if (Config->MapFile.empty())
    return;

  std::error_code EC;
  raw_fd_ostream OS(Config->MapFile, EC, sys::fs::F_None);
  if (EC)
    error("cannot open " + Config->MapFile + ": " + EC.message());
  else
    writeMapFile2<ELFT>(OS, OutputSections);
}

template void elf::writeMapFile<ELF32LE>(ArrayRef<OutputSectionBase *>);
template void elf::writeMapFile<ELF32BE>(ArrayRef<OutputSectionBase *>);
template void elf::writeMapFile<ELF64LE>(ArrayRef<OutputSectionBase *>);
template void elf::writeMapFile<ELF64BE>(ArrayRef<OutputSectionBase *>);
