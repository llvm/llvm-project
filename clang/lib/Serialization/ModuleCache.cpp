//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Serialization/ModuleCache.h"

#include "clang/Serialization/InMemoryModuleCache.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/LockFileManager.h"
#include "llvm/Support/Path.h"

using namespace clang;

namespace {
class CrossProcessModuleCache : public ModuleCache {
  InMemoryModuleCache InMemory;

public:
  void prepareForGetLock(StringRef ModuleFilename) override {
    // FIXME: Do this in LockFileManager and only if the directory doesn't
    // exist.
    StringRef Dir = llvm::sys::path::parent_path(ModuleFilename);
    llvm::sys::fs::create_directories(Dir);
  }

  std::unique_ptr<llvm::AdvisoryLock>
  getLock(StringRef ModuleFilename) override {
    return std::make_unique<llvm::LockFileManager>(ModuleFilename);
  }

  InMemoryModuleCache &getInMemoryModuleCache() override { return InMemory; }
  const InMemoryModuleCache &getInMemoryModuleCache() const override {
    return InMemory;
  }
};
} // namespace

IntrusiveRefCntPtr<ModuleCache> clang::createCrossProcessModuleCache() {
  return llvm::makeIntrusiveRefCnt<CrossProcessModuleCache>();
}
