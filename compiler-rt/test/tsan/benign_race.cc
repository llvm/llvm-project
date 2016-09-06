// RUN: %clang_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
#include "test.h"

int Global;
int WTFGlobal;

extern "C" {
void AnnotateBenignRaceSized(const char *f, int l,
                             void *mem, unsigned int size, const char *desc);
void WTFAnnotateBenignRaceSized(const char *f, int l,
                                void *mem, unsigned int size,
                                const char *desc);
}


void *Thread(void *x) {
  Global = 42;
  WTFGlobal = 142;
  barrier_wait(&barrier);
  return 0;
}

int main() {
  barrier_init(&barrier, 2);
  AnnotateBenignRaceSized(__FILE__, __LINE__,
                          &Global, sizeof(Global), "Race on Global");
  WTFAnnotateBenignRaceSized(__FILE__, __LINE__,
                             &WTFGlobal, sizeof(WTFGlobal),
                             "Race on WTFGlobal");
  pthread_t t;
  pthread_create(&t, 0, Thread, 0);
  barrier_wait(&barrier);
  Global = 43;
  WTFGlobal = 143;
  pthread_join(t, 0);
  fprintf(stderr, "OK\n");
}

// CHECK-NOT: WARNING: ThreadSanitizer: data race
