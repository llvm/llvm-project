// RUN: %clangxx_tsan -O1 %s -o %t && %deflake %run %t 2>&1 | FileCheck %s
#include "test.h"
#include <string.h>

int y[4], z[4];

void *MemMoveThread(void *a) {
  memmove((int*)a, z, 16);
  barrier_wait(&barrier);
  return NULL;
}

void *MemSetThread(void *a) {
  barrier_wait(&barrier);
  memset((int*)a, 0, 16);
  return NULL;
}

int main() {
  barrier_init(&barrier, 2);
  pthread_t t[2];
  // Race on y between memmove and memset
  pthread_create(&t[0], NULL, MemMoveThread, y);
  pthread_create(&t[1], NULL, MemSetThread, y);
  pthread_join(t[0], NULL);
  pthread_join(t[1], NULL);

  fprintf(stderr, "PASS\n");
  return 0;
}

// CHECK: WARNING: ThreadSanitizer: data race
// CHECK:   #0 {{.*}}memset
// CHECK:   #{{[12]}} MemSetThread
// CHECK:  Previous write
// CHECK:   #0 {{.*mem(cpy|move)}}
// CHECK:   #{{[12]}} MemMoveThread
