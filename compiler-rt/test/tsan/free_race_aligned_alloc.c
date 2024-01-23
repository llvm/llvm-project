// RUN: %clang_tsan -O1 %s -o %t -undefined dynamic_lookup 
// RUN: %deflake %run %t | FileCheck %s 

#include "test.h"

#ifdef __APPLE__
#include <os/availability.h>

// Allow compilation with pre-aligned-alloc SDKs
API_AVAILABLE(macos(10.15), ios(13.0), tvos(13.0), watchos(6.0))
void *aligned_alloc(size_t alignment, size_t size);
#else
#include <stdlib.h>
#endif

int *mem;
pthread_mutex_t mtx;

void *Thread1(void *x) {
  pthread_mutex_lock(&mtx);
  free(mem);
  pthread_mutex_unlock(&mtx);
  barrier_wait(&barrier);
  return NULL;
}

__attribute__((noinline)) void *Thread2(void *x) {
  barrier_wait(&barrier);
  pthread_mutex_lock(&mtx);
  mem[0] = 42;
  pthread_mutex_unlock(&mtx);
  return NULL;
}

int main() {
  if (aligned_alloc == NULL) {
    fprintf(stderr, "Done.\n");
    return 0;
  }

  barrier_init(&barrier, 2);
  mem = (int*)aligned_alloc(8, 8);
  pthread_mutex_init(&mtx, 0);
  pthread_t t;
  pthread_create(&t, NULL, Thread1, NULL);
  Thread2(0);
  pthread_join(t, NULL);
  pthread_mutex_destroy(&mtx);
  return 0;
}

// CHECK: WARNING: ThreadSanitizer: heap-use-after-free
// CHECK:   Write of size 4 at {{.*}} by main thread{{.*}}:
// CHECK:     #0 Thread2
// CHECK:     #1 main
// CHECK:   Previous write of size 8 at {{.*}} by thread T1{{.*}}:
// CHECK:     #0 free
// CHECK:     #{{(1|2)}} Thread1
// CHECK: SUMMARY: ThreadSanitizer: heap-use-after-free{{.*}}Thread2
