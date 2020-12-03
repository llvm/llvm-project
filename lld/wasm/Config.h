//===- Config.h -------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_WASM_CONFIG_H
#define LLD_WASM_CONFIG_H

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/BinaryFormat/Wasm.h"
#include "llvm/Support/CachePruning.h"

namespace lld {
namespace wasm {

// For --unresolved-symbols.
// The `ImportFuncs` mode is an additional mode that corresponds to the
// --allow-undefined flag which turns undefined functions in imports
// as opposed ed to Ignore or Warn which turn them into unreachables.
enum class UnresolvedPolicy { ReportError, Warn, Ignore, ImportFuncs };

// This struct contains the global configuration for the linker.
// Most fields are direct mapping from the command line options
// and such fields have the same name as the corresponding options.
// Most fields are initialized by the driver.
struct Configuration {
  bool bsymbolic;
  bool checkFeatures;
  bool compressRelocations;
  bool demangle;
  bool disableVerify;
  bool experimentalPic;
  bool emitRelocs;
  bool exportAll;
  bool exportDynamic;
  bool exportTable;
  bool growableTable;
  bool gcSections;
  bool importMemory;
  bool sharedMemory;
  bool importTable;
  llvm::Optional<bool> is64;
  bool mergeDataSegments;
  bool pie;
  bool printGcSections;
  bool relocatable;
  bool saveTemps;
  bool shared;
  bool stripAll;
  bool stripDebug;
  bool stackFirst;
  bool trace;
  uint64_t globalBase;
  uint64_t initialMemory;
  uint64_t maxMemory;
  uint64_t zStackSize;
  unsigned ltoPartitions;
  unsigned ltoo;
  unsigned optimize;
  llvm::StringRef thinLTOJobs;
  bool ltoNewPassManager;
  bool ltoDebugPassManager;
  UnresolvedPolicy unresolvedSymbols;

  llvm::StringRef entry;
  llvm::StringRef mapFile;
  llvm::StringRef outputFile;
  llvm::StringRef thinLTOCacheDir;

  llvm::StringSet<> allowUndefinedSymbols;
  llvm::StringSet<> exportedSymbols;
  std::vector<llvm::StringRef> searchPaths;
  llvm::CachePruningPolicy thinLTOCachePolicy;
  llvm::Optional<std::vector<std::string>> features;

  // The following config options do not directly correspond to any
  // particualr command line options.

  // True if we are creating position-independent code.
  bool isPic;

  // The table offset at which to place function addresses.  We reserve zero
  // for the null function pointer.  This gets set to 1 for executables and 0
  // for shared libraries (since they always added to a dynamic offset at
  // runtime).
  uint32_t tableBase = 0;
};

// The only instance of Configuration struct.
extern Configuration *config;

} // namespace wasm
} // namespace lld

#endif
