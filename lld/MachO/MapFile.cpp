//===- MapFile.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the -map option, which maps address ranges to their
// respective contents, plus the input file these contents were originally from.
// The contents (typically symbols) are listed in address order. Dead-stripped
// contents are included as well.
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
#include "ConcatOutputSection.h"
#include "Config.h"
#include "InputFiles.h"
#include "InputSection.h"
#include "OutputSegment.h"
#include "Symbols.h"
#include "SyntheticSections.h"
#include "Target.h"
#include "lld/Common/ErrorHandler.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Parallel.h"
#include "llvm/Support/TimeProfiler.h"

using namespace llvm;
using namespace llvm::sys;
using namespace lld;
using namespace lld::macho;

struct CStringInfo {
  uint32_t fileIndex;
  StringRef str;
};

struct MapInfo {
  SmallVector<InputFile *> files;
  SmallVector<Defined *> deadSymbols;
  DenseMap<const OutputSection *,
           SmallVector<std::pair<uint64_t /*addr*/, CStringInfo>>>
      liveCStringsForSection;
  SmallVector<CStringInfo> deadCStrings;
};

static MapInfo gatherMapInfo() {
  MapInfo info;
  for (InputFile *file : inputFiles)
    if (isa<ObjFile>(file) || isa<BitcodeFile>(file)) {
      uint32_t fileIndex = info.files.size() + 1;
      bool isReferencedFile = false;

      // Gather the dead symbols. We don't have to bother with the live ones
      // because we will pick them up as we iterate over the OutputSections
      // later.
      for (Symbol *sym : file->symbols) {
        if (auto *d = dyn_cast_or_null<Defined>(sym))
          // Only emit the prevailing definition of a symbol. Also, don't emit
          // the symbol if it is part of a cstring section (we use the literal
          // value instead, similar to ld64)
          if (d->isec && d->getFile() == file &&
              !isa<CStringInputSection>(d->isec)) {
            isReferencedFile = true;
            if (!d->isLive())
              info.deadSymbols.push_back(d);
          }
      }

      // Gather all the cstrings (both live and dead). A CString(Output)Section
      // doesn't provide us a way of figuring out which InputSections its
      // cstring contents came from, so we need to build up that mapping here.
      for (const Section *sec : file->sections) {
        for (const Subsection &subsec : sec->subsections) {
          if (auto isec = dyn_cast<CStringInputSection>(subsec.isec)) {
            auto &liveCStrings = info.liveCStringsForSection[isec->parent];
            for (const auto &[i, piece] : llvm::enumerate(isec->pieces)) {
              if (piece.live)
                liveCStrings.push_back({isec->parent->addr + piece.outSecOff,
                                        {fileIndex, isec->getStringRef(i)}});
              else
                info.deadCStrings.push_back({fileIndex, isec->getStringRef(i)});
              isReferencedFile = true;
            }
          } else {
            break;
          }
        }
      }

      if (isReferencedFile)
        info.files.push_back(file);
    }

  // cstrings are not stored in sorted order in their OutputSections, so we sort
  // them here.
  for (auto &liveCStrings : info.liveCStringsForSection)
    parallelSort(liveCStrings.second, [](const auto &p1, const auto &p2) {
      return p1.first < p2.first;
    });
  return info;
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

  os << format("# Path: %s\n", config->outputFile.str().c_str());
  os << format("# Arch: %s\n",
               getArchitectureName(config->arch()).str().c_str());

  MapInfo info = gatherMapInfo();

  os << "# Object files:\n";
  os << format("[%3u] %s\n", 0, (const char *)"linker synthesized");
  uint32_t fileIndex = 1;
  DenseMap<lld::macho::InputFile *, uint32_t> readerToFileOrdinal;
  for (InputFile *file : info.files) {
    os << format("[%3u] %s\n", fileIndex, file->getName().str().c_str());
    readerToFileOrdinal[file] = fileIndex++;
  }

  os << "# Sections:\n";
  os << "# Address\tSize    \tSegment\tSection\n";
  for (OutputSegment *seg : outputSegments)
    for (OutputSection *osec : seg->getSections()) {
      if (osec->isHidden())
        continue;

      os << format("0x%08llX\t0x%08llX\t%s\t%s\n", osec->addr, osec->getSize(),
                   seg->name.str().c_str(), osec->name.str().c_str());
    }

  os << "# Symbols:\n";
  os << "# Address\tSize    \tFile  Name\n";
  for (const OutputSegment *seg : outputSegments) {
    for (const OutputSection *osec : seg->getSections()) {
      if (auto *concatOsec = dyn_cast<ConcatOutputSection>(osec)) {
        for (const InputSection *isec : concatOsec->inputs) {
          for (Defined *sym : isec->symbols)
            os << format("0x%08llX\t0x%08llX\t[%3u] %s\n", sym->getVA(),
                         sym->size, readerToFileOrdinal[sym->getFile()],
                         sym->getName().str().data());
        }
      } else if (osec == in.cStringSection || osec == in.objcMethnameSection) {
        const auto &liveCStrings = info.liveCStringsForSection.lookup(osec);
        uint64_t lastAddr = 0; // strings will never start at address 0, so this
                               // is a sentinel value
        for (const auto &[addr, info] : liveCStrings) {
          uint64_t size = 0;
          if (addr != lastAddr)
            size = info.str.size() + 1; // include null terminator
          lastAddr = addr;
          os << format("0x%08llX\t0x%08llX\t[%3u] literal string: ", addr, size,
                       info.fileIndex);
          os.write_escaped(info.str) << "\n";
        }
      }
      // TODO print other synthetic sections
    }
  }

  if (config->deadStrip) {
    os << "# Dead Stripped Symbols:\n";
    os << "#        \tSize    \tFile  Name\n";
    for (Defined *sym : info.deadSymbols) {
      assert(!sym->isLive());
      os << format("<<dead>>\t0x%08llX\t[%3u] %s\n", sym->size,
                   readerToFileOrdinal[sym->getFile()],
                   sym->getName().str().data());
    }
    for (CStringInfo &cstrInfo : info.deadCStrings) {
      os << format("<<dead>>\t0x%08llX\t[%3u] literal string: ",
                   cstrInfo.str.size() + 1, cstrInfo.fileIndex);
      os.write_escaped(cstrInfo.str) << "\n";
    }
  }
}
