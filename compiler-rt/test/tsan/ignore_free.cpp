// RUN: %clangxx_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
#include "test.h"

extern "C" {
void AnnotateIgnoreReadsBegin(const char *f, int l);
void AnnotateIgnoreReadsEnd(const char *f, int l);
void AnnotateIgnoreWritesBegin(const char *f, int l);
void AnnotateIgnoreWritesEnd(const char *f, int l);
}

void *Thread(void *p) {
  *(int*)p = 42;
  barrier_wait(&barrier);
  return 0;
}

int main() {
  barrier_init(&barrier, 2);
  int *p = new int(0);
  pthread_t t;
  pthread_create(&t, 0, Thread, p);
  barrier_wait(&barrier);
  AnnotateIgnoreReadsBegin(__FILE__, __LINE__);
  AnnotateIgnoreWritesBegin(__FILE__, __LINE__);
  free(p);
  AnnotateIgnoreReadsEnd(__FILE__, __LINE__);
  AnnotateIgnoreWritesEnd(__FILE__, __LINE__);
  pthread_join(t, 0);
  fprintf(stderr, "OK\n");
  return 0;
}

// CHECK-NOT: WARNING: ThreadSanitizer: data race
// CHECK: OK
