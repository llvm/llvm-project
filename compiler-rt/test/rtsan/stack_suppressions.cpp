// RUN: %clangxx -fsanitize=realtime %s -o %t
// RUN: %env_rtsan_opts=halt_on_error=false %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-NOSUPPRESSIONS
// RUN: %env_rtsan_opts=suppressions='%s.supp':print_stats_on_exit=true not %run %t 2>&1 | FileCheck %s
// UNSUPPORTED: ios

// Intent: Ensure that suppressions work as intended

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <atomic>
#include <vector>

std::atomic<int> cas_atomic{0};

void *MallocViolation() { return malloc(10); }

void VectorViolations() {
  // All of these should be suppressed by *vector*
  std::vector<int> v(10);
  v.resize(20);
  v.clear();
  v.resize(0);
  v.push_back(1);
  v.reserve(10);
}

void BlockFunc() [[clang::blocking]] {
  int expected = 0;
  while (!cas_atomic.compare_exchange_weak(expected, 1)) {
    expected = cas_atomic.load();
  }
}

void *process() [[clang::nonblocking]] {
  void *ptr = MallocViolation(); // Suppressed call-stack-contains
  VectorViolations();            // Suppressed call-stack-contains with regex
  BlockFunc();                   // Suppressed function-name-matches
  free(ptr);                     // Suppressed function-name-matches

  // This is the one that should abort the program
  // Everything else is suppressed
  usleep(1);

  return ptr;
}

int main() {
  process();
  return 0;
}

// CHECK-NOT: failed to open suppressions file
// CHECK: Intercepted call to real-time unsafe function
// CHECK-SAME: usleep

// CHECK-NOT: Intercepted call to real-time unsafe function
// CHECK-NOT: malloc
// CHECK-NOT: vector
// CHECK-NOT: free
// CHECK-NOT: BlockFunc

// CHECK: RealtimeSanitizer exit stats:
// CHECK: Suppression count: 7

// CHECK-NOSUPPRESSIONS: malloc
// CHECK-NOSUPPRESSIONS: vector
// CHECK-NOSUPPRESSIONS: free
// CHECK-NOSUPPRESSIONS: BlockFunc
// CHECK-NOSUPPRESSIONS: usleep
