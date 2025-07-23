//===- SectionPriorities.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_MACHO_SECTION_PRIORITIES_H
#define LLD_MACHO_SECTION_PRIORITIES_H

#include "InputSection.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"

namespace lld::macho {

using SectionPair = std::pair<const InputSection *, const InputSection *>;
using StringPiecePair = std::pair<CStringInputSection *, size_t>;

class PriorityBuilder {
public:
  // Reads every input section's call graph profile, and combines them into
  // callGraphProfile. If an order file is present, any edges where one or both
  // of the vertices are specified in the order file are discarded.
  void extractCallGraphProfile();

  // Reads the order file at `path` into config->priorities.
  //
  // An order file has one entry per line, in the following format:
  //
  //   <cpu>:<object file>:[<symbol name> | CStringEntryPrefix <cstring hash>]
  //
  // <cpu> and <object file> are optional.
  // If not specified, then that entry tries to match either,
  //
  // 1) any symbol of the <symbol name>;
  // Parsing this format is not quite straightforward because the symbol name
  // itself can contain colons, so when encountering a colon, we consider the
  // preceding characters to decide if it can be a valid CPU type or file path.
  // If a symbol is matched by multiple entries, then it takes the
  // lowest-ordered entry (the one nearest to the front of the list.)
  //
  // or 2) any cstring literal with the given hash, if the entry has the
  // CStringEntryPrefix prefix defined below in the file. <cstring hash> is the
  // hash of cstring literal content.
  //
  // Cstring literals are not symbolized, we can't identify them by name
  // However, cstrings are deduplicated, hence unique, so we use the hash of
  // the content of cstring literals to identify them and assign priority to it.
  // We use the same hash as used in StringPiece, i.e. 31 bit:
  // xxh3_64bits(string) & 0x7fffffff
  //
  // The file can also have line comments that start with '#'.
  void parseOrderFile(StringRef path);

  // Returns layout priorities for some or all input sections. Sections are laid
  // out in decreasing order; that is, a higher priority section will be closer
  // to the beginning of its output section.
  //
  // If either an order file or a call graph profile are present, this is used
  // as the source of priorities. If both are present, the order file takes
  // precedence, but the call graph profile is still used for symbols that don't
  // appear in the order file. If neither is present, an empty map is returned.
  //
  // Each section gets assigned the priority of the highest-priority symbol it
  // contains.
  llvm::DenseMap<const InputSection *, int> buildInputSectionPriorities();
  std::vector<StringPiecePair>
      buildCStringPriorities(ArrayRef<CStringInputSection *>);

private:
  // The symbol with the smallest priority should be ordered first in the output
  // section (modulo input section contiguity constraints).
  struct SymbolPriorityEntry {
    // The priority given to a matching symbol, regardless of which object file
    // it originated from.
    int anyObjectFile = 0;
    // The priority given to a matching symbol from a particular object file.
    llvm::DenseMap<llvm::StringRef, int> objectFiles;
  };
  const llvm::StringRef CStringEntryPrefix = "CSTR;";

  std::optional<int> getSymbolPriority(const Defined *sym);
  std::optional<int> getSymbolOrCStringPriority(const StringRef key,
                                                InputFile *f);
  llvm::DenseMap<llvm::StringRef, SymbolPriorityEntry> priorities;
  llvm::MapVector<SectionPair, uint64_t> callGraphProfile;
};

extern PriorityBuilder priorityBuilder;
} // namespace lld::macho

#endif
