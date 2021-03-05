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
#include "llvm/Support/VersionTuple.h"
#include "llvm/TextAPI/MachO/Architecture.h"
#include "llvm/TextAPI/MachO/Platform.h"
#include "llvm/TextAPI/MachO/Target.h"

#include <vector>

namespace lld {
namespace macho {

class Symbol;
struct SymbolPriorityEntry;

using NamePair = std::pair<llvm::StringRef, llvm::StringRef>;
using SectionRenameMap = llvm::DenseMap<NamePair, NamePair>;
using SegmentRenameMap = llvm::DenseMap<llvm::StringRef, llvm::StringRef>;

struct PlatformInfo {
  llvm::VersionTuple minimum;
  llvm::VersionTuple sdk;
};

enum class NamespaceKind {
  twolevel,
  flat,
};

enum class UndefinedSymbolTreatment {
  unknown,
  error,
  warning,
  suppress,
  dynamic_lookup,
};

struct Configuration {
  Symbol *entry;
  bool hasReexports = false;
  bool allLoad = false;
  bool forceLoadObjC = false;
  bool staticLink = false;
  bool implicitDylibs = false;
  bool isPic = false;
  bool headerPadMaxInstallNames = false;
  bool ltoNewPassManager = LLVM_ENABLE_NEW_PASS_MANAGER;
  bool printEachFile = false;
  bool printWhyLoad = false;
  bool searchDylibsFirst = false;
  bool saveTemps = false;
  bool adhocCodesign = false;
  uint32_t headerPad;
  uint32_t dylibCompatibilityVersion = 0;
  uint32_t dylibCurrentVersion = 0;
  llvm::StringRef installName;
  llvm::StringRef outputFile;
  llvm::StringRef ltoObjPath;
  bool demangle = false;
  llvm::MachO::Target target;
  PlatformInfo platformInfo;
  NamespaceKind namespaceKind = NamespaceKind::twolevel;
  UndefinedSymbolTreatment undefinedSymbolTreatment =
      UndefinedSymbolTreatment::error;
  llvm::MachO::HeaderFileType outputType;
  std::vector<llvm::StringRef> systemLibraryRoots;
  std::vector<llvm::StringRef> librarySearchPaths;
  std::vector<llvm::StringRef> frameworkSearchPaths;
  std::vector<llvm::StringRef> runtimePaths;
  std::vector<Symbol *> explicitUndefineds;
  llvm::DenseMap<llvm::StringRef, SymbolPriorityEntry> priorities;
  SectionRenameMap sectionRenameMap;
  SegmentRenameMap segmentRenameMap;
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
