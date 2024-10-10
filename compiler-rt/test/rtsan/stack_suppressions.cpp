// RUN: %clangxx -fsanitize=realtime %s -o %t
// RUN: %env_rtsan_opts=suppressions='%s.supp' not %run %t 2>&1 | FileCheck %s
// UNSUPPORTED: ios

// Intent: Ensure that suppressions work as intended

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <vector>

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

void BlockFunc() [[clang::blocking]] { usleep(1); }

void *process() [[clang::nonblocking]] {
  void *ptr = MallocViolation();
  VectorViolations();
  BlockFunc();
  free(ptr);

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
