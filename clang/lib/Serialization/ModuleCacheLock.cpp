#include "clang/Serialization/ModuleCacheLock.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

using namespace clang;

namespace {
struct ModuleCacheFileLockManager : ModuleCacheLockManager {
  llvm::LockFileManager Lock;

  ModuleCacheFileLockManager(StringRef ModuleFilename) : Lock(ModuleFilename) {}

  operator LockResult() const override {
    switch (Lock) {
    case llvm::LockFileManager::LFS_Owned:
      return LockResult::Owned;
    case llvm::LockFileManager::LFS_Shared:
      return LockResult::Shared;
    case llvm::LockFileManager::LFS_Error:
      return LockResult::Error;
    }
  }

  WaitForUnlockResult waitForUnlock() override {
    switch (Lock.waitForUnlock()) {
    case llvm::LockFileManager::Res_Success:
      return WaitForUnlockResult::Success;
    case llvm::LockFileManager::Res_OwnerDied:
      return WaitForUnlockResult::OwnerDied;
    case llvm::LockFileManager::Res_Timeout:
      return WaitForUnlockResult::Timeout;
    }
  }

  void unsafeRemoveLock() override { Lock.unsafeRemoveLockFile(); }

  std::string getErrorMessage() const override {
    return Lock.getErrorMessage();
  }
};

struct ModuleCacheFileLock : ModuleCacheLock {
  void prepareLock(StringRef ModuleFilename) override {
    // FIXME: have LockFileManager return an error_code so that we can
    // avoid the mkdir when the directory already exists.
    StringRef Dir = llvm::sys::path::parent_path(ModuleFilename);
    llvm::sys::fs::create_directories(Dir);
  }

  std::unique_ptr<ModuleCacheLockManager>
  tryLock(StringRef ModuleFilename) override {
    return std::make_unique<ModuleCacheFileLockManager>(ModuleFilename);
  }
};
} // namespace

std::shared_ptr<ModuleCacheLock> clang::getModuleCacheFileLock() {
  return std::make_unique<ModuleCacheFileLock>();
}
