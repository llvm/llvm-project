//===- comgr-cache.h - Comgr Cache implementation -------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef COMGR_CACHE_H
#define COMGR_CACHE_H

#include "amd_comgr.h"
#include "comgr-cache-command.h"

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/CachePruning.h>
#include <llvm/Support/MemoryBuffer.h>

#include <functional>
#include <memory>

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace COMGR {
class CommandCache {
  std::string CacheDir;
  llvm::CachePruningPolicy Policy;

  CommandCache(llvm::StringRef CacheDir,
               const llvm::CachePruningPolicy &Policy);

  static std::optional<llvm::CachePruningPolicy>
  getPolicyFromEnv(llvm::raw_ostream &LogS);

public:
  static std::unique_ptr<CommandCache> get(llvm::raw_ostream &);

  ~CommandCache();
  void prune();

  /// Checks if the Command C is cached.
  /// If it is the case, it replaces its output and logs its error-stream.
  /// Otherwise it executes C through the callback Execute
  amd_comgr_status_t execute(CachedCommandAdaptor &C, llvm::raw_ostream &LogS);
};
} // namespace COMGR

#endif
