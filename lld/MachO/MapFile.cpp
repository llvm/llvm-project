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

using SymbolMapTy = DenseMap<const InputSection *, SmallVector<Defined *, 4>>;

// Returns a map from sections to their symbols.
static SymbolMapTy getSectionSyms(ArrayRef<Defined *> syms) {
  SymbolMapTy ret;
  for (Defined *dr : syms)
    ret[dr->isec].push_back(dr);

  // Sort symbols by address. We want to print out symbols in the
  // order in the output file rather than the order they appeared
  // in the input files.
  for (auto &it : ret)
    llvm::stable_sort(it.second, [](Defined *a, Defined *b) {
      return a->getVA() < b->getVA();
    });
  return ret;
}

// Returns a list of all symbols that we want to print out.
static std::vector<Defined *> getSymbols() {
  std::vector<Defined *> v;
  for (InputFile *file : inputFiles)
    if (isa<ObjFile>(file))
      for (Symbol *sym : file->symbols) {
        if (auto *d = dyn_cast_or_null<Defined>(sym))
          if (d->isLive() && d->isec && d->getFile() == file) {
            assert(!d->isec->isCoalescedWeak() &&
                   "file->symbols should store resolved symbols");
            v.push_back(d);
          }
      }
  return v;
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

  // Collect symbol info that we want to print out.
  std::vector<Defined *> syms = getSymbols();
  SymbolMapTy sectionSyms = getSectionSyms(syms);
  DenseMap<Symbol *, std::string> symStr = getSymbolStrings(syms);

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
  os << "# Symbols:\n";
  os << "# Address\t    File  Name\n";
  for (InputSection *isec : inputSections) {
    auto symsIt = sectionSyms.find(isec);
    assert(!isec->shouldOmitFromOutput() || (symsIt == sectionSyms.end()));
    if (symsIt == sectionSyms.end())
      continue;
    for (Symbol *sym : symsIt->second) {
      os << format("0x%08llX\t[%3u] %s\n", sym->getVA(),
                   readerToFileOrdinal[sym->getFile()], symStr[sym].c_str());
    }
  }

  // TODO: when we implement -dead_strip, we should dump dead stripped symbols
}
