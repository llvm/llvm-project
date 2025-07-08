#pragma once
#include "rsan_defs.hpp"

//class __tsan::ThreadState;

namespace Robustness{
class SANITIZER_MUTEX FakeMutex : public Mutex {
 public:
  explicit constexpr FakeMutex(__tsan::MutexType type = __tsan::MutexUnchecked)
      : Mutex(type) {}

  void Lock() SANITIZER_ACQUIRE() { }

  bool TryLock() SANITIZER_TRY_ACQUIRE(true) { return true; }

  void Unlock() SANITIZER_RELEASE() { }

  void ReadLock() SANITIZER_ACQUIRE_SHARED() { }

  void ReadUnlock() SANITIZER_RELEASE_SHARED() { }

  void CheckWriteLocked() const SANITIZER_CHECK_LOCKED() { }

  void CheckLocked() const SANITIZER_CHECK_LOCKED() {}

  void CheckReadLocked() const SANITIZER_CHECK_LOCKED() { }


  //FakeMutex(LinkerInitialized) = delete;
  FakeMutex(const FakeMutex &) = delete;
  void operator=(const FakeMutex &) = delete;
};
} // namespace Robustness
