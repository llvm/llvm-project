//===- Config.h -------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_MACHO_CONFIG_H
#define LLD_MACHO_CONFIG_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/MachO.h"
#include "llvm/TextAPI/MachO/Architecture.h"

#include <vector>

namespace lld {
namespace macho {

class Symbol;
struct SymbolPriorityEntry;

struct Configuration {
  Symbol *entry;
  bool hasReexports = false;
  llvm::StringRef installName;
  llvm::StringRef outputFile;
  llvm::MachO::Architecture arch;
  llvm::MachO::HeaderFileType outputType;
  std::vector<llvm::StringRef> librarySearchPaths;
  std::vector<llvm::StringRef> frameworkSearchPaths;
  llvm::DenseMap<llvm::StringRef, SymbolPriorityEntry> priorities;
};

// The symbol with the highest priority should be ordered first in the output
// section (modulo input section contiguity constraints). Using priority
// (highest first) instead of order (lowest first) has the convenient property
// that the default-constructed zero priority -- for symbols/sections without a
// user-defined order -- naturally ends up putting them at the end of the
// output.
struct SymbolPriorityEntry {
  // The priority given to a matching symbol, regardless of which object file
  // it originated from.
  size_t anyObjectFile = 0;
  // The priority given to a matching symbol from a particular object file.
  llvm::DenseMap<llvm::StringRef, size_t> objectFiles;
};

extern Configuration *config;

} // namespace macho
} // namespace lld

#endif
