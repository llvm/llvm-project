// RUN: %clangxx_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
// UNSUPPORTED: darwin

#include <stdio.h>
#include <thread>

int shared = 0;

void worker() {
  flockfile(stdout);
  shared++;
  funlockfile(stdout);
}

int main() {
  std::thread t1(worker), t2(worker);
  t1.join();
  t2.join();
}

// CHECK-NOT: ThreadSanitizer: data race
// CHECK: ThreadSanitizer: no issues found
