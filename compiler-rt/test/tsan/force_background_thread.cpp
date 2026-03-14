// RUN: %clangxx_tsan -O1 %s -o %t
// RUN: %deflake %env_tsan_opts=force_background_thread=0:verbosity=1:memory_limit_mb=1000 %run %t 2>&1 | FileCheck %s --implicit-check-not "memory flush check"
// RUN: %deflake %env_tsan_opts=force_background_thread=1:verbosity=1:memory_limit_mb=1000 %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK,THREAD
// RUN: %deflake %env_tsan_opts=force_background_thread=0:verbosity=1:memory_limit_mb=1000 %run %t 1 2>&1 | FileCheck %s --check-prefixes=CHECK,THREAD

// Fails with: objc[99984]: task_restartable_ranges_register failed (result 0x2e: (os/kern) service not supported)
// UNSUPPORTED: darwin

#include "test.h"

void *Thread(void *a) { return nullptr; }

int main(int argc, char *argv[]) {
  if (argc > 1) {
    pthread_t t;
    pthread_create(&t, nullptr, Thread, nullptr);
    void *p;
    pthread_join(t, &p);
  }
  sleep(3);
  return 1;
}

// CHECK: Running under ThreadSanitizer
// THREAD: ThreadSanitizer: memory flush check
