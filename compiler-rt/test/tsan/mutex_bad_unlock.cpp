// RUN: %clangxx_tsan -O1 %s -o %t && %deflake %run %t | FileCheck %s
#include "test.h"

int main() {
  int m = 0;
  AnnotateRWLockReleased(__FILE__, __LINE__, &m, 1);
  return 0;
}

// CHECK: WARNING: ThreadSanitizer: unlock of an unlocked mutex (or by a wrong thread)
// CHECK:     #0 AnnotateRWLockReleased
// CHECK:     #1 main
// CHECK: Location is stack of main thread.
// CHECK:   Mutex {{.*}} created at:
// CHECK:     #0 AnnotateRWLockReleased
// CHECK:     #1 main
// CHECK: SUMMARY: ThreadSanitizer: unlock of an unlocked mutex (or by a wrong thread)

