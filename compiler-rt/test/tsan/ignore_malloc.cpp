// RUN: %clangxx_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
#include "test.h"

extern "C" {
void AnnotateIgnoreReadsBegin(const char *f, int l);
void AnnotateIgnoreReadsEnd(const char *f, int l);
void AnnotateIgnoreWritesBegin(const char *f, int l);
void AnnotateIgnoreWritesEnd(const char *f, int l);
}

int *g;

void *Thread(void *a) {
  int *p = 0;
  while ((p = __atomic_load_n(&g, __ATOMIC_RELAXED)) == 0)
    usleep(100);  // spin-wait
  *p = 42;
  return 0;
}

int main() {
  pthread_t t;
  pthread_create(&t, 0, Thread, 0);
  AnnotateIgnoreWritesBegin(__FILE__, __LINE__);
  int *p = new int(0);
  AnnotateIgnoreWritesEnd(__FILE__, __LINE__);
  __atomic_store_n(&g, p, __ATOMIC_RELAXED);
  pthread_join(t, 0);
  delete p;
  fprintf(stderr, "OK\n");
  return 0;
}

// CHECK-NOT: WARNING: ThreadSanitizer: data race
// CHECK: OK
