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
//   Address  Size     Align Out     In      Symbol
//   00201000 00000015     4 .text
//   00201000 0000000e     4         test.o:.text
//   0020100e 00000000     0                 local
//   00201005 00000000     0                 f(int)
//
//===----------------------------------------------------------------------===//

#include "MapFile.h"
#include "InputFiles.h"
#include "Strings.h"
#include "SymbolTable.h"
#include "Threads.h"

#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::object;

using namespace lld;
using namespace lld::elf;

namespace {
template <class ELFT> class PrettyPrinter {
public:
  PrettyPrinter();
  void print(raw_ostream &OS, ArrayRef<OutputSection *> OutputSections);

private:
  void writeInputSection(raw_ostream &OS, const InputSection *IS);

  // Maps sections to their symbols.
  DenseMap<const SectionBase *, SmallVector<DefinedRegular *, 4>> Symbols;

  // Contains a string like this
  //
  //   0020100e 00000000     0                         f(int)
  //
  // for each symbol.
  DenseMap<SymbolBody *, std::string> SymStr;
};
} // namespace

// Print out the first three columns of a line.
template <class ELFT>
static void writeHeader(raw_ostream &OS, uint64_t Addr, uint64_t Size,
                        uint64_t Align) {
  int W = ELFT::Is64Bits ? 16 : 8;
  OS << format("%0*llx %0*llx %5lld ", W, Addr, W, Size, Align);
}

static std::string indent(int Depth) { return std::string(Depth * 8, ' '); }

template <class ELFT> PrettyPrinter<ELFT>::PrettyPrinter() {
  // Collect all symbols that we want to print out.
  std::vector<DefinedRegular *> Syms;
  for (elf::ObjectFile<ELFT> *File : Symtab<ELFT>::X->getObjectFiles())
    for (SymbolBody *B : File->getSymbols())
      if (B->File == File && !B->isSection())
        if (auto *Sym = dyn_cast<DefinedRegular>(B))
          if (Sym->Section)
            Syms.push_back(Sym);

  // Initialize the map from sections to their symbols.
  for (DefinedRegular *Sym : Syms)
    Symbols[Sym->Section].push_back(Sym);

  // Sort symbols by address. We want to print out symbols in the
  // order in the output file rather than the order they appeared
  // in the input files.
  for (auto &It : Symbols) {
    SmallVectorImpl<DefinedRegular *> &V = It.second;
    std::sort(V.begin(), V.end(), [](DefinedRegular *A, DefinedRegular *B) {
      return A->getVA() < B->getVA();
    });
  }

  // Construct a map from symbols to their stringified representations.
  // Demangling symbols is slow, so we use the parallel-for.
  std::vector<std::string> Str(Syms.size());
  parallelFor(0, Syms.size(), [&](size_t I) {
    raw_string_ostream OS(Str[I]);
    writeHeader<ELFT>(OS, Syms[I]->getVA(), Syms[I]->template getSize<ELFT>(),
                      0);
    OS << indent(2) << toString(*Syms[I]) << '\n';
  });
  for (size_t I = 0, E = Syms.size(); I < E; ++I)
    SymStr[Syms[I]] = std::move(Str[I]);
}

template <class ELFT>
void PrettyPrinter<ELFT>::writeInputSection(raw_ostream &OS,
                                            const InputSection *IS) {
  // Write a line for each symbol defined in the given section.
  writeHeader<ELFT>(OS, IS->OutSec->Addr + IS->OutSecOff, IS->getSize(),
                    IS->Alignment);
  OS << indent(1) << toString(IS) << '\n';
  for (DefinedRegular *Sym : Symbols[IS])
    OS << SymStr[Sym];
}

template <class ELFT>
void PrettyPrinter<ELFT>::print(raw_ostream &OS,
                                ArrayRef<OutputSection *> OutputSections) {
  // Print out the header line.
  int W = ELFT::Is64Bits ? 16 : 8;
  OS << left_justify("Address", W) << ' ' << left_justify("Size", W)
     << " Align Out     In      Symbol\n";

  // Print out a mapfile.
  for (OutputSection *Sec : OutputSections) {
    writeHeader<ELFT>(OS, Sec->Addr, Sec->Size, Sec->Alignment);
    OS << Sec->Name << '\n';
    for (InputSection *IS : Sec->Sections)
      writeInputSection(OS, IS);
  }
}

template <class ELFT>
void elf::writeMapFile(ArrayRef<OutputSection *> OutputSections) {
  if (Config->MapFile.empty())
    return;

  std::error_code EC;
  raw_fd_ostream OS(Config->MapFile, EC, sys::fs::F_None);
  if (EC)
    error("cannot open " + Config->MapFile + ": " + EC.message());
  else
    PrettyPrinter<ELFT>().print(OS, OutputSections);
}

template void elf::writeMapFile<ELF32LE>(ArrayRef<OutputSection *>);
template void elf::writeMapFile<ELF32BE>(ArrayRef<OutputSection *>);
template void elf::writeMapFile<ELF64LE>(ArrayRef<OutputSection *>);
template void elf::writeMapFile<ELF64BE>(ArrayRef<OutputSection *>);
