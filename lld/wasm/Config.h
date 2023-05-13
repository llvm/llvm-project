//===- Config.h -------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_WASM_CONFIG_H
#define LLD_WASM_CONFIG_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/BinaryFormat/Wasm.h"
#include "llvm/Support/CachePruning.h"
#include <optional>

namespace llvm::CodeGenOpt {
enum Level : int;
} // namespace llvm::CodeGenOpt

namespace lld::wasm {

class InputFile;
class Symbol;

// For --unresolved-symbols.
enum class UnresolvedPolicy { ReportError, Warn, Ignore, ImportDynamic };

// For --build-id.
enum class BuildIdKind { None, Fast, Sha1, Hexstring, Uuid };

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
  bool extendedConst;
  bool growableTable;
  bool gcSections;
  std::optional<std::pair<llvm::StringRef, llvm::StringRef>> memoryImport;
  std::optional<llvm::StringRef> memoryExport;
  bool sharedMemory;
  bool importTable;
  bool importUndefined;
  std::optional<bool> is64;
  bool mergeDataSegments;
  bool pie;
  bool printGcSections;
  bool relocatable;
  bool saveTemps;
  bool shared;
  bool stripAll;
  bool stripDebug;
  bool stackFirst;
  bool isStatic = false;
  bool trace;
  uint64_t globalBase;
  uint64_t initialMemory;
  uint64_t maxMemory;
  uint64_t zStackSize;
  unsigned ltoPartitions;
  unsigned ltoo;
  llvm::CodeGenOpt::Level ltoCgo;
  unsigned optimize;
  llvm::StringRef thinLTOJobs;
  bool ltoDebugPassManager;
  UnresolvedPolicy unresolvedSymbols;
  BuildIdKind buildId = BuildIdKind::None;

  llvm::StringRef entry;
  llvm::StringRef mapFile;
  llvm::StringRef outputFile;
  llvm::StringRef thinLTOCacheDir;
  llvm::StringRef whyExtract;

  llvm::StringSet<> allowUndefinedSymbols;
  llvm::StringSet<> exportedSymbols;
  std::vector<llvm::StringRef> requiredExports;
  llvm::SmallVector<llvm::StringRef, 0> searchPaths;
  llvm::CachePruningPolicy thinLTOCachePolicy;
  std::optional<std::vector<std::string>> features;
  std::optional<std::vector<std::string>> extraFeatures;
  llvm::SmallVector<uint8_t, 0> buildIdVector;

  // The following config options do not directly correspond to any
  // particular command line options, and should probably be moved to separate
  // Ctx struct as in ELF/Config.h

  // True if we are creating position-independent code.
  bool isPic;

  // True if we have an MVP input that uses __indirect_function_table and which
  // requires it to be allocated to table number 0.
  bool legacyFunctionTable = false;

  // The table offset at which to place function addresses.  We reserve zero
  // for the null function pointer.  This gets set to 1 for executables and 0
  // for shared libraries (since they always added to a dynamic offset at
  // runtime).
  uint32_t tableBase = 0;

  // Will be set to true if bss data segments should be emitted. In most cases
  // this is not necessary.
  bool emitBssSegments = false;

  // A tuple of (reference, extractedFile, sym). Used by --why-extract=.
  llvm::SmallVector<std::tuple<std::string, const InputFile *, const Symbol &>,
                    0>
      whyExtractRecords;
};

// The only instance of Configuration struct.
extern Configuration *config;

} // namespace lld::wasm

#endif
