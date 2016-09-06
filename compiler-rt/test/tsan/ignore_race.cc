// RUN: %clang_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
#include "test.h"

int Global;

extern "C" void AnnotateIgnoreWritesBegin(const char *f, int l);
extern "C" void AnnotateIgnoreWritesEnd(const char *f, int l);
extern "C" void AnnotateIgnoreReadsBegin(const char *f, int l);
extern "C" void AnnotateIgnoreReadsEnd(const char *f, int l);

void *Thread(void *x) {
  AnnotateIgnoreWritesBegin(__FILE__, __LINE__);
  AnnotateIgnoreReadsBegin(__FILE__, __LINE__);
  Global = 42;
  AnnotateIgnoreReadsEnd(__FILE__, __LINE__);
  AnnotateIgnoreWritesEnd(__FILE__, __LINE__);
  barrier_wait(&barrier);
  return 0;
}

int main() {
  barrier_init(&barrier, 2);
  pthread_t t;
  pthread_create(&t, 0, Thread, 0);
  barrier_wait(&barrier);
  Global = 43;
  pthread_join(t, 0);
  fprintf(stderr, "OK\n");
}

// CHECK-NOT: WARNING: ThreadSanitizer: data race
