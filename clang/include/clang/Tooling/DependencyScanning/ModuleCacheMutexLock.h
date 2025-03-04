#ifndef LLVM_CLANG_TOOLING_DEPENDENCYSCANNING_MODULECACHEMUTEXLOCK_H
#define LLVM_CLANG_TOOLING_DEPENDENCYSCANNING_MODULECACHEMUTEXLOCK_H

#include "clang/Serialization/ModuleCacheLock.h"
#include "llvm/ADT/StringMap.h"

#include <condition_variable>

namespace clang {
namespace tooling {
namespace dependencies {
struct ModuleCacheMutexWrapper {
  std::mutex Mutex;
  std::condition_variable CondVar;
  bool Done = false;

  ModuleCacheMutexWrapper() = default;
};

struct ModuleCacheMutexes {
  std::mutex Mutex;
  llvm::StringMap<std::shared_ptr<ModuleCacheMutexWrapper>> Map;
};

std::shared_ptr<ModuleCacheLock>
getModuleCacheMutexLock(ModuleCacheMutexes &Mutexes);
} // namespace dependencies
} // namespace tooling
} // namespace clang

#endif
