// RUN: %clangxx_tsan -O1 %s -o %t && %deflake %run %t | FileCheck %s
#include <pthread.h>
#include <stdio.h>

extern "C" void AnnotateIgnoreReadsBegin(const char *f, int l);

void *Thread(void *x) {
  AnnotateIgnoreReadsBegin("", 0);
  return 0;
}

int main() {
  pthread_t t;
  pthread_create(&t, 0, Thread, 0);
  pthread_join(t, 0);
}

// CHECK: ThreadSanitizer: thread T1  finished with ignores enabled, created at:
// CHECK:     #0 pthread_create
// CHECK:     #1 main
// CHECK:   Ignore was enabled at:
// CHECK:     #0 AnnotateIgnoreReadsBegin
// CHECK:     #1 Thread

