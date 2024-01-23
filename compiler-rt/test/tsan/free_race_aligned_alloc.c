// RUN: %clang_tsan -O1 %s -o %t -undefined dynamic_lookup
// RUN: %deflake %run %t | FileCheck %s 

#include "test.h"

#include <stdlib.h>

#if defined(__cplusplus) && (__cplusplus >= 201703L)
#define HAVE_ALIGNED_ALLOC 1
#endif

#if defined(__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__) && \
    __ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__ < 101500
#define HAVE_ALIGNED_ALLOC 0 
#else
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

  barrier_init(&barrier, 2);
#if HAVE_ALIGNED_ALLOC
  mem = (int*)aligned_alloc(8, 8);
#else
  mem = (int*)malloc(8);
#endif
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
