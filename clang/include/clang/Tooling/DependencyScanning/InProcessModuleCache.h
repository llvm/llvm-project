//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_DEPENDENCYSCANNING_INPROCESSMODULECACHE_H
#define LLVM_CLANG_TOOLING_DEPENDENCYSCANNING_INPROCESSMODULECACHE_H

#include "clang/Serialization/ModuleCache.h"
#include "llvm/ADT/StringMap.h"

#include <mutex>
#include <shared_mutex>

namespace clang {
namespace tooling {
namespace dependencies {
struct ModuleCacheEntry {
  std::shared_mutex CompilationMutex;
  std::atomic<std::time_t> Timestamp = 0;
};

struct ModuleCacheEntries {
  std::mutex Mutex;
  llvm::StringMap<std::unique_ptr<ModuleCacheEntry>> Map;
};

IntrusiveRefCntPtr<ModuleCache>
makeInProcessModuleCache(ModuleCacheEntries &Entries);
} // namespace dependencies
} // namespace tooling
} // namespace clang

#endif
