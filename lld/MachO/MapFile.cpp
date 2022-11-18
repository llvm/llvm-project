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
// symbols:
//
// # Path: test
// # Arch: x86_84
// # Object files:
// [  0] linker synthesized
// [  1] a.o
// # Sections:
// # Address    Size       Segment  Section
// 0x1000005C0  0x0000004C __TEXT   __text
// # Symbols:
// # Address    Size       File  Name
// 0x1000005C0  0x00000001 [  1] _main
// # Dead Stripped Symbols:
// #            Size       File  Name
// <<dead>>     0x00000001 [  1] _foo
//
//===----------------------------------------------------------------------===//

#include "MapFile.h"
#include "Config.h"
#include "InputFiles.h"
#include "InputSection.h"
#include "OutputSection.h"
#include "OutputSegment.h"
#include "Symbols.h"
#include "SyntheticSections.h"
#include "Target.h"
#include "lld/Common/ErrorHandler.h"
#include "llvm/Support/Parallel.h"
#include "llvm/Support/TimeProfiler.h"

using namespace llvm;
using namespace llvm::sys;
using namespace lld;
using namespace lld::macho;

struct MapInfo {
  SmallVector<InputFile *> files;
  SmallVector<Defined *> liveSymbols;
  SmallVector<Defined *> deadSymbols;
};

static MapInfo gatherMapInfo() {
  MapInfo info;
  for (InputFile *file : inputFiles)
    if (isa<ObjFile>(file) || isa<BitcodeFile>(file)) {
      bool hasEmittedSymbol = false;
      for (Symbol *sym : file->symbols) {
        if (auto *d = dyn_cast_or_null<Defined>(sym))
          if (d->isec && d->getFile() == file) {
            if (d->isLive()) {
              assert(!shouldOmitFromOutput(d->isec));
              info.liveSymbols.push_back(d);
            } else {
              info.deadSymbols.push_back(d);
            }
            hasEmittedSymbol = true;
          }
      }
      if (hasEmittedSymbol)
        info.files.push_back(file);
    }
  parallelSort(info.liveSymbols.begin(), info.liveSymbols.end(),
               [](Defined *a, Defined *b) { return a->getVA() < b->getVA(); });
  return info;
}

// Construct a map from symbols to their stringified representations.
// Demangling symbols (which is what toString() does) is slow, so
// we do that in batch using parallel-for.
static DenseMap<Symbol *, std::string>
getSymbolStrings(ArrayRef<Defined *> syms) {
  std::vector<std::string> str(syms.size());
  parallelFor(0, syms.size(), [&](size_t i) {
    raw_string_ostream os(str[i]);
    Defined *sym = syms[i];

    switch (sym->isec->kind()) {
    case InputSection::CStringLiteralKind: {
      // Output "literal string: <string literal>"
      const auto *isec = cast<CStringInputSection>(sym->isec);
      const StringPiece &piece = isec->getStringPiece(sym->value);
      assert(
          sym->value == piece.inSecOff &&
          "We expect symbols to always point to the start of a StringPiece.");
      StringRef str = isec->getStringRef(&piece - &(*isec->pieces.begin()));
      (os << "literal string: ").write_escaped(str);
      break;
    }
    case InputSection::ConcatKind:
    case InputSection::WordLiteralKind:
      os << toString(*sym);
    }
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

  MapInfo info = gatherMapInfo();

  // Dump table of object files.
  os << "# Object files:\n";
  os << format("[%3u] %s\n", 0, (const char *)"linker synthesized");
  uint32_t fileIndex = 1;
  DenseMap<lld::macho::InputFile *, uint32_t> readerToFileOrdinal;
  for (InputFile *file : info.files) {
    os << format("[%3u] %s\n", fileIndex, file->getName().str().c_str());
    readerToFileOrdinal[file] = fileIndex++;
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
  DenseMap<Symbol *, std::string> liveSymbolStrings =
      getSymbolStrings(info.liveSymbols);
  os << "# Symbols:\n";
  os << "# Address\tSize    \tFile  Name\n";
  for (Defined *sym : info.liveSymbols) {
    assert(sym->isLive());
    os << format("0x%08llX\t0x%08llX\t[%3u] %s\n", sym->getVA(), sym->size,
                 readerToFileOrdinal[sym->getFile()],
                 liveSymbolStrings[sym].c_str());
  }

  if (config->deadStrip) {
    DenseMap<Symbol *, std::string> deadSymbolStrings =
        getSymbolStrings(info.deadSymbols);
    os << "# Dead Stripped Symbols:\n";
    os << "#        \tSize    \tFile  Name\n";
    for (Defined *sym : info.deadSymbols) {
      assert(!sym->isLive());
      os << format("<<dead>>\t0x%08llX\t[%3u] %s\n", sym->size,
                   readerToFileOrdinal[sym->getFile()],
                   deadSymbolStrings[sym].c_str());
    }
  }
}
