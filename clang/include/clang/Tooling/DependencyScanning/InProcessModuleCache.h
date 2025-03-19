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

#include <shared_mutex>

namespace clang {
namespace tooling {
namespace dependencies {
struct ModuleCacheMutexes {
  std::mutex Mutex;
  llvm::StringMap<std::unique_ptr<std::shared_mutex>> Map;
};

IntrusiveRefCntPtr<ModuleCache>
makeInProcessModuleCache(ModuleCacheMutexes &Mutexes);
} // namespace dependencies
} // namespace tooling
} // namespace clang

#endif
