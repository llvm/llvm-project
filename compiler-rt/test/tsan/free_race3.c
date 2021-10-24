// RUN: %clang_tsan -O1 %s -o %t && %deflake %run %t | FileCheck %s

#include "test.h"

int *mem;

void *Thread(void *x) {
  mem[0] = 42;
  barrier_wait(&barrier);
  return NULL;
}

int main() {
  barrier_init(&barrier, 2);
  mem = (int*)malloc(100);
  pthread_t t;
  pthread_create(&t, 0, Thread, 0);
  barrier_wait(&barrier);
  free(mem);
  pthread_join(t, NULL);
  return 0;
}

// CHECK: WARNING: ThreadSanitizer: data race
// CHECK:   Write of size 8 at {{.*}} by main thread{{.*}}:
// CHECK:     #0 free
// CHECK:     #1 main
// CHECK:   Previous write of size 4 at {{.*}} by thread T1{{.*}}:
// CHECK:     #0 Thread
