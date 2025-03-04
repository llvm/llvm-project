#include "clang/Tooling/DependencyScanning/ModuleCacheMutexLock.h"

using namespace clang;
using namespace tooling;
using namespace dependencies;

namespace {
struct ModuleCacheMutexLockManager : ModuleCacheLockManager {
  ModuleCacheMutexes &Mutexes;
  StringRef Filename;

  std::shared_ptr<ModuleCacheMutexWrapper> MutexWrapper;
  bool Owning;

  ModuleCacheMutexLockManager(ModuleCacheMutexes &Mutexes, StringRef Filename)
      : Mutexes(Mutexes), Filename(Filename) {
    Owning = false;
    {
      std::lock_guard Lock(Mutexes.Mutex);
      auto &MutexWrapperInMap = Mutexes.Map[Filename];
      if (!MutexWrapperInMap) {
        MutexWrapperInMap = std::make_shared<ModuleCacheMutexWrapper>();
        Owning = true;
      }
      // Increment the reference count of the mutex here in the critical section
      // to guarantee that it's kept alive even when another thread removes it
      // via \c unsafeRemoveLock().
      MutexWrapper = MutexWrapperInMap;
    }

    if (Owning)
      MutexWrapper->Mutex.lock();
  }

  ~ModuleCacheMutexLockManager() override {
    if (Owning) {
      MutexWrapper->Done = true;
      MutexWrapper->CondVar.notify_all();
      MutexWrapper->Mutex.unlock();
    }
  }

  operator LockResult() const override {
    return Owning ? LockResult::Owned : LockResult::Shared;
  }

  WaitForUnlockResult waitForUnlock() override {
    assert(!Owning);
    std::unique_lock Lock(MutexWrapper->Mutex);
    bool Done = MutexWrapper->CondVar.wait_for(
        Lock, std::chrono::seconds(90), [&] { return MutexWrapper->Done; });
    return Done ? WaitForUnlockResult::Success : WaitForUnlockResult::Timeout;
  }

  void unsafeRemoveLock() override {
    std::lock_guard Lock(Mutexes.Mutex);
    Mutexes.Map[Filename].reset();
  }

  std::string getErrorMessage() const override {
    llvm_unreachable("ModuleCacheMutexLockManager cannot fail");
  }
};

struct ModuleCacheMutexLock : ModuleCacheLock {
  ModuleCacheMutexes &Mutexes;

  ModuleCacheMutexLock(ModuleCacheMutexes &Mutexes) : Mutexes(Mutexes) {}

  void prepareLock(StringRef Filename) override {}

  std::unique_ptr<ModuleCacheLockManager> tryLock(StringRef Filename) override {
    return std::make_unique<ModuleCacheMutexLockManager>(Mutexes, Filename);
  }
};
} // namespace

std::shared_ptr<ModuleCacheLock>
dependencies::getModuleCacheMutexLock(ModuleCacheMutexes &Mutexes) {
  return std::make_shared<ModuleCacheMutexLock>(Mutexes);
}
