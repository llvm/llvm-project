//===- MapFile.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the -map option. It shows lists in order and
// hierarchically the outputFile, arch, input files, output sections and
// symbol:
//
// # Path: test
// # Arch: x86_84
// # Object files:
// [  0] linker synthesized
// [  1] a.o
// # Sections:
// # Address  Size      Segment  Section
// 0x1000005C0  0x0000004C  __TEXT  __text
// # Symbols:
// # Address  File  Name
// 0x1000005C0  [  1] _main
//
//===----------------------------------------------------------------------===//

#include "MapFile.h"
#include "Config.h"
#include "InputFiles.h"
#include "InputSection.h"
#include "OutputSection.h"
#include "OutputSegment.h"
#include "Symbols.h"
#include "Target.h"
#include "llvm/Support/Parallel.h"
#include "llvm/Support/TimeProfiler.h"

using namespace llvm;
using namespace llvm::sys;
using namespace lld;
using namespace lld::macho;

using Symbols = std::vector<Defined *>;
// Returns a pair where the left element is a container of all live Symbols and
// the right element is a container of all dead symbols.
static std::pair<Symbols, Symbols> getSymbols() {
  Symbols liveSymbols, deadSymbols;
  for (InputFile *file : inputFiles)
    if (isa<ObjFile>(file))
      for (Symbol *sym : file->symbols)
        if (auto *d = dyn_cast_or_null<Defined>(sym))
          if (d->isec && d->getFile() == file) {
            if (d->isLive()) {
              assert(!shouldOmitFromOutput(d->isec));
              liveSymbols.push_back(d);
            } else {
              deadSymbols.push_back(d);
            }
          }
  parallelSort(liveSymbols.begin(), liveSymbols.end(),
               [](Defined *a, Defined *b) {
                 return a->getVA() != b->getVA() ? a->getVA() < b->getVA()
                                                 : a->getName() < b->getName();
               });
  parallelSort(
      deadSymbols.begin(), deadSymbols.end(),
      [](Defined *a, Defined *b) { return a->getName() < b->getName(); });
  return {std::move(liveSymbols), std::move(deadSymbols)};
}

// Construct a map from symbols to their stringified representations.
// Demangling symbols (which is what toString() does) is slow, so
// we do that in batch using parallel-for.
static DenseMap<Symbol *, std::string>
getSymbolStrings(ArrayRef<Defined *> syms) {
  std::vector<std::string> str(syms.size());
  parallelForEachN(0, syms.size(), [&](size_t i) {
    raw_string_ostream os(str[i]);
    os << toString(*syms[i]);
  });

  DenseMap<Symbol *, std::string> ret;
  for (size_t i = 0, e = syms.size(); i < e; ++i)
    ret[syms[i]] = std::move(str[i]);
  return ret;
}

void macho::writeMapFile() {
  if (config->mapFile.empty())
    return;

  TimeTraceScope timeScope("Write map file");

  // Open a map file for writing.
  std::error_code ec;
  raw_fd_ostream os(config->mapFile, ec, sys::fs::OF_None);
  if (ec) {
    error("cannot open " + config->mapFile + ": " + ec.message());
    return;
  }

  // Dump output path.
  os << format("# Path: %s\n", config->outputFile.str().c_str());

  // Dump output architecture.
  os << format("# Arch: %s\n",
               getArchitectureName(config->arch()).str().c_str());

  // Dump table of object files.
  os << "# Object files:\n";
  os << format("[%3u] %s\n", 0, (const char *)"linker synthesized");
  uint32_t fileIndex = 1;
  DenseMap<lld::macho::InputFile *, uint32_t> readerToFileOrdinal;
  for (InputFile *file : inputFiles) {
    if (isa<ObjFile>(file)) {
      os << format("[%3u] %s\n", fileIndex, file->getName().str().c_str());
      readerToFileOrdinal[file] = fileIndex++;
    }
  }

  // Dump table of sections
  os << "# Sections:\n";
  os << "# Address\tSize    \tSegment\tSection\n";
  for (OutputSegment *seg : outputSegments)
    for (OutputSection *osec : seg->getSections()) {
      if (osec->isHidden())
        continue;

      os << format("0x%08llX\t0x%08llX\t%s\t%s\n", osec->addr, osec->getSize(),
                   seg->name.str().c_str(), osec->name.str().c_str());
    }

  // Dump table of symbols
  Symbols liveSymbols, deadSymbols;
  std::tie(liveSymbols, deadSymbols) = getSymbols();

  DenseMap<Symbol *, std::string> liveSymbolStrings =
      getSymbolStrings(liveSymbols);
  os << "# Symbols:\n";
  os << "# Address\t    File  Name\n";
  for (Symbol *sym : liveSymbols) {
    assert(sym->isLive());
    os << format("0x%08llX\t[%3u] %s\n", sym->getVA(),
                 readerToFileOrdinal[sym->getFile()],
                 liveSymbolStrings[sym].c_str());
  }

  if (config->deadStrip) {
    DenseMap<Symbol *, std::string> deadSymbolStrings =
        getSymbolStrings(deadSymbols);
    os << "# Dead Stripped Symbols:\n";
    os << "# Address\t    File  Name\n";
    for (Symbol *sym : deadSymbols) {
      assert(!sym->isLive());
      os << format("<<dead>>\t[%3u] %s\n", readerToFileOrdinal[sym->getFile()],
                   deadSymbolStrings[sym].c_str());
    }
  }
}
