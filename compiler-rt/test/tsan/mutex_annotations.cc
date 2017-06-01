// RUN: %clangxx_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
#include "test.h"

// Test that a linker-initialized mutex can be created/destroyed while in use.

// Stub for testing, just invokes annotations.
// Meant to be synchronized externally with test barrier.
class Mutex {
 public:
  void Create(bool linker_initialized = false) {
    if (linker_initialized)
      ANNOTATE_RWLOCK_CREATE_STATIC(&state_);
    else
      ANNOTATE_RWLOCK_CREATE(&state_);
  }

  void Destroy() {
    ANNOTATE_RWLOCK_DESTROY(&state_);
  }

  void Lock() {
    ANNOTATE_RWLOCK_ACQUIRED(&state_, true);
  }

  void Unlock() {
    ANNOTATE_RWLOCK_RELEASED(&state_, true);
  }

 private:
  long long state_;
};

int main() {
  Mutex m;

  m.Lock();
  m.Create(true);
  m.Unlock();

  m.Lock();
  m.Destroy();
  m.Unlock();

  fprintf(stderr, "DONE\n");
  return 0;
}

// CHECK-NOT: WARNING: ThreadSanitizer:
// CHECK: DONE
