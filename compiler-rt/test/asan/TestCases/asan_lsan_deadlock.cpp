// Test for potential deadlock in LeakSanitizer+AddressSanitizer.
// REQUIRES: leak-detection
//
// RUN: %clangxx_asan -O0 %s -o %t
// RUN: %env_asan_opts=detect_leaks=1 not %run %t 2>&1 | FileCheck %s

// FIXME: Hangs for unknown reasons on all platforms. We can re-enable it when
// its either deterministic, or we solve the deadlock between asan and lsan.
// UNSUPPORTED: true

/*
 * Purpose: Verify deadlock prevention between ASan error reporting and LSan leak checking.
 * 
 * Test Design:
 * 1. Creates contention scenario between:
 *    - ASan's error reporting (requires lock B -> lock A ordering)
 *    - LSan's leak check (requires lock A -> lock B ordering)
 * 2. Thread timing:
 *    - Main thread: Holds 'in' mutex -> Triggers LSan check (lock A then B) 
 *    - Worker thread: Triggers ASan OOB error (lock B then A via symbolization)
 * 
 * Deadlock Condition (if unfixed):
 * Circular lock dependency forms when:
 * [Main Thread] LSan: lock A -> requests lock B
 * [Worker Thread] ASan: lock B -> requests lock A
 * 
 * Success Criteria: 
 * With proper lock ordering enforcement, watchdog should NOT trigger - test exits with Asan report.
  */

#include <mutex>
#include <sanitizer/lsan_interface.h>
#include <stdio.h>
#include <thread>
#include <unistd.h>

void Watchdog() {
  // Safety mechanism: Turn infinite deadlock into finite test failure
  sleep(60);
  // Unexpected. "not" in RUN will fail if we reached here.
  _exit(0);
}

int main(int argc, char **argv) {
  int arr[1] = {0};
  std::mutex in;
  in.lock();

  std::thread w(Watchdog);
  w.detach();

  std::thread t([&]() {
    in.unlock();
    /* 
     * Provoke ASan error: ASan's error reporting acquires: 
     * 1. ASan's thread registry lock (B) during the reporting 
     * 2. dl_iterate_phdr lock (A) during symbolization
     */
    // CHECK: SUMMARY: AddressSanitizer: stack-buffer-overflow
    arr[argc] = 1; // Deliberate OOB access
  });

  in.lock();
  /* 
   * Critical section: LSan's check acquires: 
   * 1. dl_iterate_phdr lock (A)
   * 2. ASan's thread registry lock (B)
   * before Stop The World.
   */
  __lsan_do_leak_check();
  t.join();
  return 0;
}
